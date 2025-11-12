#!/usr/bin/env python3
"""
Script to populate the plants_data table with image URLs
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

# Database configuration
DB_USER = os.getenv("DB_USER", "mlops_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "mlops_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# GitHub base URLs
BASE_URL = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"


def generate_image_urls(label: str, count: int = 200) -> list:
    """Generate image URLs for a given label"""
    urls = []
    for i in range(count):
        url = f"{BASE_URL}/{label}/{i:08d}.jpg"
        urls.append({
            'url_source': url,
            'label': label,
            'url_s3': None,
            'processed': False
        })
    return urls


def populate_database():
    """Populate the database with image URLs"""
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        logger.info(f"Connected to database: {DB_NAME}")
        
        # Generate URLs for dandelions
        logger.info("Generating URLs for dandelions...")
        dandelion_urls = generate_image_urls("dandelion", 200)
        
        # Generate URLs for grass
        logger.info("Generating URLs for grass...")
        grass_urls = generate_image_urls("grass", 200)
        
        # Combine all URLs
        all_urls = dandelion_urls + grass_urls
        
        # Create DataFrame
        df = pd.DataFrame(all_urls)
        logger.info(f"Total URLs generated: {len(df)}")
        logger.info(f"Dandelions: {len(dandelion_urls)}, Grass: {len(grass_urls)}")
        
        # Insert into database
        df.to_sql('plants_data', engine, if_exists='append', index=False)
        logger.success("✅ Database populated successfully!")
        
        # Verify
        with engine.connect() as conn:
            result = conn.execute("SELECT label, COUNT(*) FROM plants_data GROUP BY label")
            for row in result:
                logger.info(f"{row[0]}: {row[1]} images")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error populating database: {e}")
        return False


def main():
    logger.info("🚀 Starting database population...")
    success = populate_database()
    
    if success:
        logger.success("✅ Database initialization completed!")
        sys.exit(0)
    else:
        logger.error("❌ Database initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
