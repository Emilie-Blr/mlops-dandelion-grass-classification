# 🌼 MLOps Project: Dandelion vs Grass Classification

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)]()

## 📝 Description

Ce projet implémente un **pipeline MLOps complet** pour la classification d'images binaire (pissenlit 🌼 vs herbe 🌿) avec :

✅ **Objectif 1-3**: Extraction, prétraitement et stockage des données (Airflow + MinIO)  
✅ **Objectif 4**: Tracking des modèles et expériences avec MLflow  
✅ **Objectif 5**: API REST pour les prédictions (FastAPI)  
✅ **Objectif 6**: Interface utilisateur interactive (Streamlit)  
🔄 **Objectif 7**: Dockerisation et déploiement Kubernetes  
🔄 **Objectif 9**: Continuous Training automatisé  
🔄 **Objectif 10**: Monitoring avec Prometheus + Grafana  

**Date limite**: Dimanche 2 novembre 2025 à minuit  
**Présentation**: 10min démo + 15min Q&A  

## 👥 Équipe (Groupe 5)

- Emilie BOULANGER
- Hugo BRAUN  
- Nehemie BIKUKA PRINCE
- Maria BOUSSA
- Sara BEN ABDELKADER

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   GitHub    │─────▶│ GitHub Actions│─────▶│  DockerHub  │
│             │      │   (CI/CD)     │      │             │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │   API    │  │ Streamlit│  │  MLflow  │  │ Airflow  ││
│  │ FastAPI  │  │  WebApp  │  │          │  │          ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┤
│  │Prometheus│  │  Grafana │  │     MinIO (S3)           │
│  └──────────┘  └──────────┘  └──────────────────────────┤
└─────────────────────────────────────────────────────────┘
```

## 📚 Stack Technologique

| Composant | Technologie |
|-----------|-------------|
| **Orchestration** | Apache Airflow |
| **ML Framework** | PyTorch / FastAI |
| **Model Registry** | MinIO (S3) |
| **ML Tracking** | MLflow |
| **Database** | PostgreSQL |
| **API** | FastAPI |
| **WebApp** | Streamlit |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus + Grafana |
| **Load Testing** | Locust |

## 🚀 Quick Start

### Prérequis
```bash
# Requis
- Docker Desktop (avec Kubernetes)
- Python 3.9+
- Git

# Optionnel  
- kubectl
- helm
```

### 🏃‍♂️ Démarrage Rapide (5 minutes)

```bash
# 1. Cloner le repository
git clone https://github.com/iblamesro/mlops-dandelion-grass-classification.git
cd MLproject

# 2. Lancer l'infrastructure avec Docker Compose
docker-compose up -d

# 3. Attendre que les services démarrent (2-3 minutes)
docker-compose ps

# 4. Lancer l'API
python3 run_api.py &

# 5. Lancer la WebApp
python3 run_webapp.py
```

**🎉 C'est prêt !**
- 🌐 WebApp: http://localhost:8501
- 📚 API Docs: http://localhost:8000/docs
- 🔬 MLflow: http://localhost:5001
- ✈️ Airflow: http://localhost:8082 (admin/admin)
- 📦 MinIO: http://localhost:9001 (minioadmin/minioadmin)

### 📦 Installation des Dépendances

```bash
# Dépendances complètes
pip install -r requirements.txt

# Dépendances API uniquement (léger)
pip install -r requirements-api.txt
```

## 🎯 Objectifs Complétés

| # | Objectif | Status | Détails |
|---|----------|--------|---------|
| 1 | Extract & preprocess data | ✅ | Airflow DAG, téléchargement depuis GitHub |
| 2 | Build classification model | ✅ | ResNet18, PyTorch, 91.59% accuracy |
| 3 | Store model on S3 | ✅ | MinIO (S3 compatible), modèle 128MB |
| 4 | Track with MLFlow | ✅ | 6 runs trackés, métriques + paramètres |
| 5 | **Develop API** | ✅ | **FastAPI, 4 endpoints, Swagger docs** |
| 6 | **Create WebApp** | ✅ | **Streamlit, upload images, visualisations** |
| 7 | Dockerize & deploy K8s | 🔄 | Dockerfiles prêts, déploiement en cours |
| 8 | Version on GitHub | ✅ | Repository public avec CI/CD |
| 9 | Retraining pipeline | 🔄 | DAG Airflow continuous training |
| 10 | Add monitoring | 🔄 | Prometheus + Grafana configurés |

**Progression**: 6/10 objectifs principaux complétés (60%)

## 📊 Utilisation Détaillée

### 1️⃣ API FastAPI (Objectif 5)

**Lancer l'API:**
```bash
python3 run_api.py
# Accessible sur http://localhost:8000
```

**Endpoints disponibles:**
- `GET /` - Informations de base
- `GET /health` - Health check
- `GET /model/info` - Informations du modèle  
- `POST /predict` - Prédiction sur image
- `GET /docs` - Documentation Swagger interactive
- `GET /metrics` - Métriques Prometheus

**Tester avec curl:**
```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/dandelion/00000001.jpg"
```

**Tester avec Python:**
```python
import requests

with open("data/dandelion/00000001.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    print(response.json())
# {'predicted_class': 'dandelion', 'confidence': 0.9876, 'prediction_time': 0.123}
```

### 2️⃣ WebApp Streamlit (Objectif 6)

**Deux versions disponibles:**

**🌟 Version PRO (Recommandée)** - Avec fonctionnalités avancées :
```bash
python3 run_webapp_pro.py
# Accessible sur http://localhost:8501
```

**Fonctionnalités PRO** :
- 🌙 **Dark Mode** - Toggle clair/sombre
- 📁 **Batch Processing** - Upload multiple d'images
- 📊 **Historique & Stats** - Sauvegarde des prédictions avec timeline
- 🎨 **UI Améliorée** - Animations CSS, gradients, hover effects
- 📈 **Visualisations Avancées** - Gauge chart, timeline, pie chart
- 💾 **Export CSV** - Sauvegarde des résultats batch
- 🔍 **Ajustements d'image** - Brightness, contrast
- 🎯 **Multi-tabs** - Différentes vues des résultats

**📖 Guide complet** : Voir `WEBAPP_PRO_GUIDE.md`

**Version Standard** - Interface basique :
```bash
python3 run_webapp.py
# Accessible sur http://localhost:8501
```

**Lancer la WebApp:**
```bash
python3 run_webapp.py
# Accessible sur http://localhost:8501
```

**Fonctionnalités:**
- 📤 Upload d'images (JPG, PNG)
- 🔮 Prédictions en temps réel
- 📊 Graphiques de confiance (Plotly)
- ⚙️ Mode local ou API backend
- 🎨 Interface responsive et moderne

**Utilisation:**
1. Ouvrir http://localhost:8501
2. Uploader une image (drag & drop ou browse)
3. Cliquer sur "🔮 Classify Image"
4. Observer le résultat avec le score de confiance

**Mode API:**
- Dans le sidebar, cocher "🌐 Use API"
- L'app utilisera l'API FastAPI au lieu du modèle local
- Tester la connexion avec "Test Connection"

### 3️⃣ MLflow Tracking (Objectif 4)

**Interface MLflow:**
```bash
# Déjà lancé avec docker-compose
# Accéder à http://localhost:5001
```

**Voir les runs:**
```bash
python scripts/check_mlflow.py
```

**Expérience**: `dandelion-grass-classification`  
**Meilleur run**: 91.59% accuracy, 91.59% F1-score

### 4️⃣ Entraînement du Modèle

**Script d'entraînement avancé:**
```bash
python scripts/train_advanced.py \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001
```

**Via Airflow:**
```bash
# Trigger le DAG de training
docker-compose exec airflow-webserver \
  airflow dags trigger model_training_pipeline
```

### 5️⃣ Pipeline de Données

**Télécharger les images:**
```bash
python scripts/download_images.py
# Télécharge depuis dandelion.csv et grass.csv
# Sauvegarde dans data/dandelion/ et data/grass/
```

**Via Airflow:**
```bash
# Trigger le DAG d'extraction
docker-compose exec airflow-webserver \
  airflow dags trigger data_extraction_pipeline
```

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration  
pytest tests/integration/

# Tests avec coverage
pytest --cov=src tests/

# Test de charge (API)
python scripts/load_test.py
```

## 📁 Structure du Projet (Optimisée)

```
MLproject/
├── src/
│   ├── api/
│   │   └── main.py              # ✅ API FastAPI complète
│   ├── webapp/
│   │   └── app_enhanced.py      # ✅ WebApp Streamlit
│   ├── training/
│   │   ├── model.py             # Architecture ResNet18
│   │   └── train.py             # Script d'entraînement + MLflow
│   ├── utils/
│   │   ├── helpers.py           # Utilitaires
│   │   └── s3_client.py         # Client MinIO/S3
│   └── config.py                # Configuration centralisée
│
├── scripts/
│   ├── check_mlflow.py          # Vérifier les runs MLflow
│   ├── download_images.py       # Télécharger les datasets
│   ├── test_api.py              # Tests automatisés API
│   ├── train_advanced.py        # Entraînement avec options
│   └── ...                      # Autres utilitaires
│
├── airflow/
│   └── dags/
│       ├── data_extraction_dag.py      # Pipeline extraction
│       └── continuous_training_dag.py  # Training automatique
│
├── models/
│   ├── best_model.pth           # ✅ Modèle entraîné (128MB)
│   └── model_metadata.json      # Métadonnées
│
├── data/
│   ├── dandelion/               # Images pissenlit (200)
│   └── grass/                   # Images herbe (200)
│
├── k8s/                         # Manifestes Kubernetes
├── monitoring/                  # Config Prometheus + Grafana
├── tests/                       # Tests unitaires & intégration
│
├── dandelion.csv               # ✅ URLs images pissenlit
├── grass.csv                   # ✅ URLs images herbe
├── docker-compose.yml          # Stack complète (dev)
├── Dockerfile.api              # Image Docker API
├── Dockerfile.webapp           # Image Docker WebApp
├── requirements.txt            # Dépendances complètes
├── requirements-api.txt        # Dépendances API minimales
├── run_api.py                  # ✅ Launcher API
├── run_webapp.py               # ✅ Launcher WebApp
├── README.md                   # 📖 Ce fichier
└── QUICKSTART.md              # Guide démarrage rapide
```

**Fichiers supprimés (optimisation):**
- ❌ `docs/` - Tout consolidé dans README
- ❌ `logs/` - Dossier vide
- ❌ `api.log` - Log vide
- ❌ `src/webapp/app.py` - Version simple (on garde enhanced)
- ❌ Fichiers `__pycache__/`, `.pyc`, `.DS_Store`

## 🎯 Résultats du Modèle

### Métriques de Performance

| Métrique | Valeur |
|----------|--------|
| **Test Accuracy** | 91.59% |
| **Test Precision** | 92.45% |
| **Test Recall** | 90.74% |
| **Test F1-Score** | 91.59% |
| **Training Time** | ~10 minutes |
| **Model Size** | 128 MB |
| **Inference Time** | ~0.1s/image |

### Architecture du Modèle
- **Base**: ResNet18 (pre-trained ImageNet)
- **Classes**: 2 (Dandelion, Grass)
- **Input Size**: 224x224 RGB
- **Parameters**: ~11M
- **Framework**: PyTorch 2.1.0

### Dataset
- **Total**: 400 images (200 dandelion + 200 grass)
- **Train/Val/Test Split**: 383/43/107
- **Source**: GitHub raw images
- **Augmentation**: Random crop, flip, rotation, color jitter

## 🔧 Choix Techniques & Justifications

### Pourquoi PyTorch ?
- ✅ Flexibilité pour l'architecture du modèle
- ✅ Excellente communauté et documentation
- ✅ Support natif MPS (Apple Silicon)
- ✅ Écosystème riche (torchvision, etc.)

### Pourquoi FastAPI ?
- ✅ Performance excellente (async/await)
- ✅ Documentation automatique (Swagger/ReDoc)
- ✅ Validation des données avec Pydantic
- ✅ Compatible avec Kubernetes
- ✅ Support WebSocket et Server-Sent Events

### Pourquoi Streamlit ?
- ✅ Développement rapide d'interfaces
- ✅ Pas besoin de HTML/CSS/JS
- ✅ Composants riches (charts, widgets)
- ✅ Hot reload pendant le développement
- ✅ Déploiement simple

### Pourquoi MLflow ?
- ✅ Standard de l'industrie
- ✅ Tracking des expériences
- ✅ Model registry intégré
- ✅ Support multi-frameworks
- ✅ API simple et intuitive

### Pourquoi Airflow ?
- ✅ Standard pour orchestration de pipelines
- ✅ UI intuitive et complète
- ✅ Extensibilité avec operators
- ✅ Monitoring intégré
- ✅ Gestion des dépendances

### Pourquoi ResNet18 ?
- ✅ Architecture éprouvée pour classification
- ✅ Poids pré-entraînés (transfer learning)
- ✅ Bon équilibre performance/taille
- ✅ Rapide à l'inférence

## 📈 Monitoring & Observabilité

### Prometheus (http://localhost:9091)
Métriques collectées:
- Requêtes API (count, latency, errors)
- Utilisation CPU/Mémoire
- Prédictions par classe
- Temps d'inférence

### Grafana (http://localhost:3001)
Dashboards disponibles:
- 📊 API Performance
- � Model Metrics  
- ✈️ Airflow Pipeline Status
- 🖥️ Infrastructure Health

### Logs
- API: Loguru avec rotation
- Airflow: Logs par DAG/Task
- Docker: `docker-compose logs -f [service]`

## 🔄 Continuous Training

**DAG**: `continuous_training_dag.py`

**Triggers automatiques:**
1. ✅ Nouvelles données (>100 images)
2. ✅ Schedule hebdomadaire (dimanche 2h)
3. ✅ Performance < seuil (90% accuracy)

**Pipeline:**
1. Extraction nouvelles données
2. Validation et preprocessing
3. Re-entraînement du modèle
4. Évaluation sur test set
5. Si meilleur: save + deploy
6. Notification (Slack/Email)

## 🐳 Docker & Kubernetes

### Images Docker

```bash
# Build images
docker build -f Dockerfile.api -t mlops-api:latest .
docker build -f Dockerfile.webapp -t mlops-webapp:latest .

# Push to DockerHub
docker push [username]/mlops-api:latest
docker push [username]/mlops-webapp:latest
```

### Déploiement Kubernetes

```bash
# Créer namespace
kubectl create namespace mlops

# Déployer services
kubectl apply -f k8s/

# Vérifier
kubectl get pods -n mlops
kubectl get svc -n mlops

# Port forward pour tests locaux
kubectl port-forward svc/api-service 8000:8000 -n mlops
kubectl port-forward svc/webapp-service 8501:8501 -n mlops
```

## 🔐 Sécurité & Best Practices

- ✅ Variables d'environnement pour secrets
- ✅ .gitignore pour fichiers sensibles
- ✅ Validation des inputs (Pydantic)
- ✅ Rate limiting sur API
- ✅ Health checks pour tous les services
- ✅ Logs sans données sensibles
- ✅ CORS configuré correctement
- ✅ Dépendances à jour

## 🚨 Troubleshooting

### L'API ne démarre pas
```bash
# Vérifier les logs
docker-compose logs api

# Vérifier le port
lsof -i :8000

# Relancer
python3 run_api.py
```

### La WebApp ne se connecte pas à l'API
```bash
# Vérifier que l'API tourne
curl http://localhost:8000/health

# Dans la WebApp: activer mode API et tester connexion
```

### MLflow ne montre pas les runs
```bash
# Vérifier MLflow
curl http://localhost:5001/health

# Vérifier les runs
python scripts/check_mlflow.py

# Redémarrer MLflow
docker-compose restart mlflow
```

### Problème avec Docker
```bash
# Nettoyer tout
docker-compose down -v
docker system prune -a

# Relancer
docker-compose up -d
```

## 📚 Documentation Complémentaire

- **QUICKSTART.md**: Guide de démarrage rapide (5min)
- **Swagger API**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5001
- **Airflow UI**: http://localhost:8082

## 🤝 Contribution & Workflow Git

```bash
# 1. Créer une branche
git checkout -b feature/nouvelle-fonctionnalite

# 2. Faire les modifications
git add .
git commit -m "feat: ajouter nouvelle fonctionnalité"

# 3. Push
git push origin feature/nouvelle-fonctionnalite

# 4. Créer une Pull Request sur GitHub
```

**Convention de commits:**
- `feat:` Nouvelle fonctionnalité
- `fix:` Correction de bug
- `docs:` Documentation
- `refactor:` Refactoring code
- `test:` Ajout de tests
- `ci:` CI/CD

## 📧 Contact & Support

**Projet**: MLOps Dandelion vs Grass Classification  
**École**: Albert School  
**Enseignant**: prillard.martin@gmail.com  
**Deadline**: 2 novembre 2025, minuit  
**Repository**: https://github.com/iblamesro/mlops-dandelion-grass-classification  

## 📄 Licence

Ce projet est réalisé dans le cadre d'un projet éducatif pour Albert School.

---

**⭐ N'oubliez pas de star le repo si ce projet vous a été utile !**
