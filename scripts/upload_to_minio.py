#!/usr/bin/env python3
"""
Upload downloaded images to MinIO bucket
"""

import os
from pathlib import Path
from minio import Minio
from tqdm import tqdm

# MinIO configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops-data"

# Directories
DATA_DIR = Path("data")
DANDELION_DIR = DATA_DIR / "dandelion"
GRASS_DIR = DATA_DIR / "grass"


def get_minio_client():
    """Create MinIO client"""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def create_bucket_if_not_exists(client, bucket_name):
    """Create bucket if it doesn't exist"""
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"✅ Bucket '{bucket_name}' created")
    else:
        print(f"✅ Bucket '{bucket_name}' already exists")


def upload_images(client, local_dir, bucket_name, prefix):
    """Upload all images from local directory to MinIO"""
    image_files = list(local_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"⚠️  No images found in {local_dir}")
        return 0
    
    print(f"\n📤 Uploading {len(image_files)} images from {local_dir.name}...")
    
    success_count = 0
    for image_path in tqdm(image_files, desc=f"Uploading {prefix}"):
        try:
            object_name = f"{prefix}/{image_path.name}"
            client.fput_object(
                bucket_name,
                object_name,
                str(image_path),
                content_type="image/jpeg"
            )
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to upload {image_path.name}: {e}")
    
    print(f"✅ {prefix}: {success_count}/{len(image_files)} images uploaded")
    return success_count


def main():
    print("🚀 Starting upload to MinIO...")
    
    # Initialize MinIO client
    try:
        client = get_minio_client()
        print("✅ Connected to MinIO")
    except Exception as e:
        print(f"❌ Failed to connect to MinIO: {e}")
        print("💡 Make sure Docker containers are running (docker-compose up -d)")
        return
    
    # Create bucket
    create_bucket_if_not_exists(client, BUCKET_NAME)
    
    # Upload dandelion images
    # ✅ MODIFIÉ: "dandelion" au lieu de "images/dandelion"
    dandelion_count = upload_images(
        client, 
        DANDELION_DIR, 
        BUCKET_NAME, 
        "dandelion"
    )
    
    # Upload grass images
    # ✅ MODIFIÉ: "grass" au lieu de "images/grass"
    grass_count = upload_images(
        client, 
        GRASS_DIR, 
        BUCKET_NAME, 
        "grass"
    )
    
    # Summary
    total = dandelion_count + grass_count
    print("\n" + "="*50)
    print("📊 UPLOAD SUMMARY")
    print("="*50)
    print(f"Total uploaded: {total} images")
    print(f"Dandelions: {dandelion_count}")
    print(f"Grass: {grass_count}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"MinIO endpoint: http://{MINIO_ENDPOINT}")
    print("="*50)


if __name__ == "__main__":
    main()