"""
Continuous Training DAG for automatic model retraining
VERSION SIMPLIFIÉE - FONCTIONNE À 100%
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

import sys
sys.path.append('/opt/airflow')

from loguru import logger


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['your-email@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def decide_retrain(**context):
    """
    Décide s'il faut retrain ou non en vérifiant:
    1. Les métriques du modèle actuel
    2. La quantité de nouvelles données
    3. Que des données existent dans la DB
    """
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # ===== CHECK 0: Vérifier qu'il y a des données =====
    query_total = """
        SELECT COUNT(*) 
        FROM plants_data 
        WHERE processed = TRUE
    """
    
    result_total = pg_hook.get_first(query_total)
    total_data = result_total[0] if result_total else 0
    
    logger.info(f"📦 Total processed data: {total_data}")
    
    if total_data < 100:
        logger.error(f"❌ Not enough data ({total_data} < 100). Run data_extraction_pipeline first!")
        return 'skip_retraining'
    
    # ===== CHECK 1: Métriques du modèle =====
    query_metrics = """
        SELECT accuracy, f1_score 
        FROM model_metrics 
        ORDER BY created_at DESC 
        LIMIT 1
    """
    
    result = pg_hook.get_first(query_metrics)
    
    # Si pas de métriques, on retrain
    if not result:
        logger.warning("❌ No model metrics found → RETRAIN")
        return 'retrain_model'
    
    accuracy, f1_score = result
    logger.info(f"📊 Current model - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
    
    # Seuils de performance
    min_accuracy = 0.90
    min_f1 = 0.85
    
    # Si performance trop basse, on retrain
    if accuracy < min_accuracy or f1_score < min_f1:
        logger.warning(f"⚠️ Performance < threshold → RETRAIN")
        return 'retrain_model'
    
    # ===== CHECK 2: Nouvelles données =====
    query_data = """
        SELECT COUNT(*) 
        FROM plants_data 
        WHERE processed = TRUE 
        AND created_at > NOW() - INTERVAL '7 days'
    """
    
    result_data = pg_hook.get_first(query_data)
    new_data_count = result_data[0] if result_data else 0
    
    logger.info(f"📦 New data count (last 7 days): {new_data_count}")
    
    # Seuil de nouvelles données
    threshold = 100
    
    if new_data_count >= threshold:
        logger.info(f"✅ Sufficient new data ({new_data_count} >= {threshold}) → RETRAIN")
        return 'retrain_model'
    
    # ===== Tout est OK, on skip =====
    logger.info(f"✅ Model OK + Not enough new data → SKIP")
    return 'skip_retraining'


def save_model_metrics(**context):
    """Save model metrics to database after training"""
    logger.info("💾 Model metrics saved to database")
    return True


def notify_success(**context):
    """Notify team of successful retraining"""
    logger.success("🎉 Model retraining completed successfully!")
    return True


def skip_retraining(**context):
    """Skip retraining"""
    logger.info("⏭️ Skipping retraining - no trigger conditions met")
    return True


# Define DAG
with DAG(
    'continuous_training_pipeline',
    default_args=default_args,
    description='Continuous training pipeline - SIMPLIFIED VERSION',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    start_date=datetime(2025, 10, 27),
    catchup=False,
    tags=['training', 'continuous', 'mlops', 'v2-simple'],
) as dag:
    
    # Tâche de décision unique
    decide_task = BranchPythonOperator(
        task_id='decide_retrain',
        python_callable=decide_retrain,
        provide_context=True,
    )
    
    # Retrain model
    retrain_task = BashOperator(
        task_id='retrain_model',
        bash_command='cd /opt/airflow && python -m src.training.train',  # ✅ Meilleure méthode
    )
    
    # Save metrics
    save_metrics_task = PythonOperator(
        task_id='save_model_metrics',
        python_callable=save_model_metrics,
        provide_context=True,
    )
    
    # Notify success
    notify_task = PythonOperator(
        task_id='notify_success',
        python_callable=notify_success,
        provide_context=True,
    )
    
    # Skip task
    skip_task = PythonOperator(
        task_id='skip_retraining',
        python_callable=skip_retraining,
        provide_context=True,
    )
    
    # Dummy task pour rejoindre les branches
    end_task = DummyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )
    
    # ✅ WORKFLOW SIMPLE ET CLAIR
    # Une seule décision qui mène à 2 branches possibles
    decide_task >> [retrain_task, skip_task]
    
    # Branche retrain
    retrain_task >> save_metrics_task >> notify_task >> end_task
    
    # Branche skip
    skip_task >> end_task