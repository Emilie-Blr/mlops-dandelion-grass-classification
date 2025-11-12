"""
Training script for the classification model
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.training.model import create_model, save_model
from src.utils.s3_client import minio_client


class PlantDataset(Dataset):
    """Custom dataset for plant images"""
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        """
        Args:
            image_paths: List of paths to images (S3 object names)
            labels: List of labels (0 or 1)
            transform: Optional transform to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image from MinIO
        image = minio_client.get_image(settings.S3_BUCKET_DATA, self.image_paths[idx])
        
        if image is None:
            # Return a blank image if download fails
            image = Image.new('RGB', (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_data_from_minio() -> Tuple[list, list]:
    """Load data paths and labels from MinIO"""
    
    logger.info("Loading data from MinIO...")
    
    image_paths = []
    labels = []
    
    # Get dandelion images - ✅ SIMPLIFIÉ (minio_client.list_objects retourne déjà des strings)
    dandelion_images = minio_client.list_objects(settings.S3_BUCKET_DATA, prefix="dandelion/")
    image_paths.extend(dandelion_images)
    labels.extend([0] * len(dandelion_images))
    
    # Get grass images - ✅ SIMPLIFIÉ (minio_client.list_objects retourne déjà des strings)
    grass_images = minio_client.list_objects(settings.S3_BUCKET_DATA, prefix="grass/")
    image_paths.extend(grass_images)
    labels.extend([1] * len(grass_images))
    
    logger.info(f"Loaded {len(dandelion_images)} dandelion images")
    logger.info(f"Loaded {len(grass_images)} grass images")
    logger.info(f"Total: {len(image_paths)} images")
    
    return image_paths, labels


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return val_loss, accuracy, precision, recall, f1


def train():
    """Main training function"""
    
    logger.info("🚀 Starting training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    image_paths, labels = load_data_from_minio()
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = PlantDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantDataset(val_paths, val_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # ✅ 0 pour éviter les problèmes de shared memory dans Docker
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0  # ✅ 0 pour éviter les problèmes de shared memory dans Docker
    )
    
    # Create model
    model = create_model(pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    
    # MLflow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": "ResNet18",
            "batch_size": settings.BATCH_SIZE,
            "learning_rate": settings.LEARNING_RATE,
            "num_epochs": settings.NUM_EPOCHS,
            "image_size": settings.IMAGE_SIZE,
            "optimizer": "Adam",
        })
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(settings.NUM_EPOCHS):
            logger.info(f"\n📊 Epoch {epoch+1}/{settings.NUM_EPOCHS}")
            
            # Train
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }, step=epoch)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            logger.info(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # ✅ Créer le dossier s'il n'existe pas
                settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                
                model_path = settings.MODELS_DIR / f"{settings.MODEL_NAME}_best.pth"
                save_model(model, str(model_path))
                logger.success(f"✅ Saved best model with accuracy: {val_acc*100:.2f}%")
                
                # Log model to MLflow
                mlflow.pytorch.log_model(model, "model")
                
                # Upload to MinIO
                minio_client.upload_file(
                    settings.S3_BUCKET_MODELS,
                    f"{settings.MODEL_NAME}_best.pth",
                    model_path
                )
        
        # Log final metrics
        mlflow.log_metric("best_val_acc", best_val_acc)
        
        logger.success(f"🎉 Training completed! Best validation accuracy: {best_val_acc*100:.2f}%")
        
        return best_val_acc


if __name__ == "__main__":
    train()