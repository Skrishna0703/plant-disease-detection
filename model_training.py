"""
Model training script for plant disease detection.
Trains a CNN model on PlantVillage dataset and saves the model and history.
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path


class PlantDiseaseModel:
    """Builds and trains a CNN model for plant disease detection."""
    
    def __init__(self, img_size=224, num_classes=None):
        """
        Initialize model parameters.
        
        Args:
            img_size: Input image size (default: 224x224)
            num_classes: Number of disease classes
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_model(self, num_classes):
        """
        Build CNN model with Conv2D, BatchNormalization, MaxPool, and Dropout layers.
        
        Args:
            num_classes: Number of output classes for Softmax
        
        Returns:
            Compiled Keras model
        """
        self.num_classes = num_classes
        
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(self.img_size, self.img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_dir, val_dir, epochs=10, batch_size=32):
        """
        Train the model using data from directories.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            zoom_range=0.15,
            shear_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data (only rescaling, no augmentation)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Store class indices for label mapping
        self.class_indices = train_generator.class_indices
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'models/plant_disease.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Number of classes: {train_generator.num_classes}")
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[early_stop, model_checkpoint],
            verbose=1
        )
        
        self.history = history
        return history
    
    def save_model(self, model_path='models/plant_disease.h5'):
        """
        Save trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_history(self, history_path='models/history.json'):
        """
        Save training history as JSON.
        
        Args:
            history_path: Path to save history
        """
        if self.history is None:
            raise ValueError("No training history to save.")
        
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = {
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'loss': [float(x) for x in self.history.history['loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Training history saved to {history_path}")
    
    def save_class_indices(self, output_path='labels.json', label_map=None):
        """
        Save class indices mapping.
        
        Args:
            output_path: Path to save class indices
            label_map: Optional dict with custom labels and remedies
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if label_map is None:
            # Create simple mapping if no label_map provided
            labels = {}
            for class_name, idx in self.class_indices.items():
                labels[class_name] = {
                    "display": class_name.replace('_', ' '),
                    "remedy": "Consult agricultural expert for treatment."
                }
        else:
            labels = label_map
        
        with open(output_path, 'w') as f:
            json.dump(labels, f, indent=4)
        
        print(f"Class indices saved to {output_path}")


def main():
    """Main training script."""
    # Paths (use split directories)
    train_dir = 'data/train_split'
    val_dir = 'data/val_split'
    
    # Verify data directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Error: Training or validation data directory not found.")
        print(f"Please create '{train_dir}' and '{val_dir}' directories with class subfolders.")
        return
    
    # Count classes
    num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    if num_classes == 0:
        print(f"Error: No class folders found in {train_dir}")
        return
    
    print(f"Found {num_classes} disease classes")
    
    # Initialize model
    model_trainer = PlantDiseaseModel(img_size=224, num_classes=num_classes)
    
    # Build model
    print("\nBuilding model...")
    model_trainer.build_model(num_classes)
    model_trainer.model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model_trainer.train(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=10,
        batch_size=32
    )
    
    # Save model and history
    print("\nSaving model and history...")
    model_trainer.save_model('models/plant_disease.h5')
    model_trainer.save_history('models/history.json')
    
    # Print final metrics
    if history:
        print("\nTraining Complete!")
        print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
