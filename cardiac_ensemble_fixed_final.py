# cardiac_ensemble_fixed_final.py
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
import cv2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib

# Add path for data loader import
sys.path.append('C:\\Alex The Great\\Project\\medai-env\\Scikit-learn\\session10')

# ---------------------- Data Pipeline Fixed ----------------------
class AdvancedDataPipelineFixed:
    def __init__(self, data_loader, target_size=(192, 192)):
        self.data_loader = data_loader
        self.target_size = target_size
    
    def prepare_data(self, samples_indices, view='2ch', phase='ED', batch_size=32, augment=True):
        """Prepare data with complete class checking"""
        frames = []
        masks = []
        
        print(f"Preparing data for {len(samples_indices)} samples ({view}, {phase})...")
        
        for idx in samples_indices:
            frame_key = f"{view}_{phase}_frame"
            mask_key = f"{view}_{phase}_mask"
            
            sample_data = self.data_loader.load_patient_data(idx)
            frame = sample_data.get(frame_key)
            mask = sample_data.get(mask_key)
            
            if frame is not None and mask is not None:
                # Resize
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                
                # Normalize frame
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                
                frames.append(frame)
                masks.append(mask)
        
        if len(frames) == 0:
            raise ValueError("No valid samples found! Check data loader.")
        
        frames = np.array(frames)[..., np.newaxis]  # channel dimension
        masks = np.array(masks)
        
        # Check class distribution
        all_mask_values = masks.flatten()
        class_counts = Counter(all_mask_values)
        print(f"Class distribution: {dict(class_counts)}")
        
        print(f"Data prepared: {frames.shape} frames, {masks.shape} masks")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((frames, masks))
        dataset = dataset.shuffle(1000)
        
        if augment:
            dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_data(self, frame, mask):
        """Data augmentation with mask structure preservation - fixed"""
        # Add dimension to mask for augmentation
        mask_expanded = tf.expand_dims(mask, axis=-1)
        
        # Left-right flip
        if tf.random.uniform(()) > 0.5:
            frame = tf.image.flip_left_right(frame)
            mask_expanded = tf.image.flip_left_right(mask_expanded)
        
        # Up-down flip
        if tf.random.uniform(()) > 0.5:
            frame = tf.image.flip_up_down(frame)
            mask_expanded = tf.image.flip_up_down(mask_expanded)
        
        # Remove extra dimension from mask
        mask_aug = tf.squeeze(mask_expanded, axis=-1)
        
        return frame, mask_aug

# ---------------------- Ensemble Model Fixed ----------------------
class EnsembleCardiacSegmentationFixed:
    def __init__(self, input_shape=(192, 192, 1), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        self.meta_learner = None
        self.model_dir = f"ensemble_cardiac_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Model directory: {self.model_dir}")
    
    def build_unet_advanced(self, name="unet_advanced"):
        """Advanced U-Net with higher capacity"""
        inputs = layers.Input(shape=self.input_shape)
        
        def conv_block(x, filters, kernel_size=3, dropout_rate=0.0):
            x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)
            return x
        
        # Encoder with higher capacity
        e1 = conv_block(inputs, 64)
        p1 = layers.MaxPooling2D((2, 2))(e1)
        
        e2 = conv_block(p1, 128)
        p2 = layers.MaxPooling2D((2, 2))(e2)
        
        e3 = conv_block(p2, 256)
        p3 = layers.MaxPooling2D((2, 2))(e3)
        
        e4 = conv_block(p3, 512, dropout_rate=0.3)
        p4 = layers.MaxPooling2D((2, 2))(e4)
        
        # Bridge
        b = conv_block(p4, 1024, dropout_rate=0.3)
        
        # Decoder
        u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
        u4 = layers.concatenate([u4, e4])
        u4 = conv_block(u4, 512)
        
        u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u4)
        u3 = layers.concatenate([u3, e3])
        u3 = conv_block(u3, 256)
        
        u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u3)
        u2 = layers.concatenate([u2, e2])
        u2 = conv_block(u2, 128)
        
        u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u2)
        u1 = layers.concatenate([u1, e1])
        u1 = conv_block(u1, 64)
        
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(u1)
        
        model = Model(inputs, outputs, name=name)
        return model
    
    def build_deep_unet_improved(self, name="deep_unet_improved"):
        """Deeper U-Net with higher capacity"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder with higher capacity
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bridge
        b = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
        b = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b)
        
        # Decoder
        u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
        u4 = layers.concatenate([u4, c4])
        u4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u4)
        u4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u4)
        
        u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u4)
        u3 = layers.concatenate([u3, c3])
        u3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        u3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u3)
        
        u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u3)
        u2 = layers.concatenate([u2, c2])
        u2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        
        u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u2)
        u1 = layers.concatenate([u1, c1])
        u1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        u1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(u1)
        
        model = Model(inputs, outputs, name=name)
        return model
    
    def compile_models(self, learning_rate=1e-3):
        """Compile models with optimal settings"""
        print("Compiling models...")
        
        self.models['unet_advanced'] = self.build_unet_advanced()
        self.models['deep_unet_improved'] = self.build_deep_unet_improved()
        
        for name, model in self.models.items():
            model.compile(
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"Compiled {name}: {model.count_params():,} parameters")
    
    def train_models(self, train_dataset, val_dataset, epochs=150):
        """Train models with advanced checkpointing"""
        histories = {}
        
        for name, model in self.models.items():
            print(f"Training {name} for {epochs} epochs...")
            
            model_save_path = os.path.join(self.model_dir, f'best_{name}.h5')
            checkpoint_path = os.path.join(self.model_dir, f'checkpoint_{name}.h5')
            
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=25,  # increased patience
                    restore_best_weights=True,
                    verbose=1,
                    mode='max'
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=15,  # increased patience
                    min_lr=1e-7,  # decreased min_lr
                    verbose=1,
                    mode='min'
                ),
                callbacks.ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                    mode='max'
                ),
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=False,
                    save_weights_only=False,
                    verbose=0,
                    save_freq='epoch'
                ),
                callbacks.TensorBoard(
                    log_dir=os.path.join(self.model_dir, f'logs_{name}'),
                    histogram_freq=1
                )
            ]
            
            try:
                print(f"Starting training for {name}...")
                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset,
                    callbacks=callbacks_list,
                    verbose=1
                )
                histories[name] = history
                
                # Final save
                model.save(model_save_path)
                print(f"FINISHED training {name}")
                
                # Show best results
                best_val_acc = max(history.history['val_accuracy'])
                best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
                print(f"Best validation accuracy for {name}: {best_val_acc:.4f} at epoch {best_epoch}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return histories
    
    def load_saved_models(self):
        """Load saved models"""
        print("Loading saved models...")
        model_files = {
            'unet_advanced': 'best_unet_advanced.h5',
            'deep_unet_improved': 'best_deep_unet_improved.h5'
        }
        
        models_loaded = 0
        for name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                print(f"Loading {name} from {model_path}")
                try:
                    self.models[name] = tf.keras.models.load_model(model_path)
                    models_loaded += 1
                    print(f"Successfully loaded {name}")
                except Exception as e:
                    print(f"Error loading {name}: {e}")
                    return False
            else:
                print(f"Saved model not found for {name}: {model_path}")
                return False
        
        return models_loaded == len(model_files)
    
    def extract_features(self, dataset):
        """Extract features for meta-learning"""
        print("Extracting features for meta-learning...")
        
        all_features = []
        all_true_labels = []
        
        # Convert dataset to numpy
        frames_list = []
        masks_list = []
        for batch in dataset:
            f, m = batch
            frames_list.append(f.numpy())
            masks_list.append(m.numpy())
        
        if not frames_list:
            print("No data in dataset!")
            return np.array([]), np.array([])
            
        frames = np.concatenate(frames_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)
        
        # Extract true labels (dominant class for each image)
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            all_true_labels.append(dominant_class)
        
        all_true_labels = np.array(all_true_labels)
        
        # Extract features from each model
        for name, model in self.models.items():
            print(f"   Processing {name}...")
            predictions = model.predict(frames, verbose=0, batch_size=16)
            
            features = []
            for pred in predictions:
                # Various features from prediction
                class_probs = np.mean(pred, axis=(0, 1))  # mean probability per class
                max_probs = np.max(pred, axis=(0, 1))     # max probability per class
                entropy = -np.sum(pred * np.log(pred + 1e-8))  # entropy
                confidence = np.mean(max_probs)  # overall confidence
                
                feature_vector = np.concatenate([
                    class_probs,    # 4 features
                    max_probs,      # 4 features  
                    [entropy],      # 1 feature
                    [confidence]    # 1 feature
                ])
                features.append(feature_vector)
            
            all_features.append(np.array(features))
        
        if not all_features:
            print("No features extracted!")
            return np.array([]), np.array([])
            
        combined_features = np.concatenate(all_features, axis=1)
        
        print(f"Features shape: {combined_features.shape}")
        print(f"Labels distribution: {Counter(all_true_labels)}")
        
        return combined_features, all_true_labels
    
    def train_meta_learner(self, train_dataset, val_dataset):
        """Train meta-learner"""
        print("Training Meta-Learner...")
        
        # Extract features
        X_train, y_train = self.extract_features(train_dataset)
        X_val, y_val = self.extract_features(val_dataset)
        
        if X_train.size == 0 or X_val.size == 0:
            print("No features for meta-learning!")
            self.meta_learner = 'averaging'
            self.meta_learner_name = 'simple_averaging'
            return 0.0
        
        # Check classes
        unique_classes = np.unique(y_train)
        print(f"Unique classes in training: {unique_classes}")
        
        if len(unique_classes) < 2:
            print("Only one class - using averaging")
            self.meta_learner = 'averaging'
            self.meta_learner_name = 'simple_averaging'
            return 0.0
        
        # Train meta-learners
        meta_learners = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,  # increased estimators
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,  # increased estimators
                learning_rate=0.1,
                max_depth=5,  # increased depth
                random_state=42
            )
        }
        
        best_score = 0
        best_learner = None
        
        for name, learner in meta_learners.items():
            print(f"Training {name}...")
            try:
                learner.fit(X_train, y_train)
                y_pred = learner.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                print(f"   {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_learner = learner
                    self.meta_learner = learner
                    self.meta_learner_name = name
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        if best_learner is not None:
            # Save meta-learner
            meta_path = os.path.join(self.model_dir, 'meta_learner.pkl')
            joblib.dump(best_learner, meta_path)
            
            feature_info = {
                'feature_dim': X_train.shape[1],
                'meta_learner': self.meta_learner_name,
                'validation_accuracy': best_score
            }
            with open(os.path.join(self.model_dir, 'feature_info.json'), 'w') as f:
                json.dump(feature_info, f, indent=4)
            
            print(f"Best meta-learner: {self.meta_learner_name} (acc: {best_score:.4f})")
        else:
            print("No meta-learner trained - using averaging")
            self.meta_learner = 'averaging'
            self.meta_learner_name = 'simple_averaging'
            best_score = 0.0
        
        return best_score
    
    def calculate_dice_scores(self, y_true, y_pred):
        """Calculate Dice scores for each class"""
        dice_scores = {}
        for class_id in range(self.num_classes):
            true_binary = (y_true == class_id).astype(np.float32)
            pred_binary = (y_pred == class_id).astype(np.float32)
            
            intersection = np.sum(true_binary * pred_binary)
            union = np.sum(true_binary) + np.sum(pred_binary)
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores[class_id] = dice
        
        return dice_scores
    
    def evaluate_complete(self, test_dataset):
        """Complete evaluation with all metrics"""
        print("Complete Evaluation...")
        
        # Evaluate individual models
        individual_results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            test_results = model.evaluate(test_dataset, verbose=0)
            individual_results[name] = {
                'loss': test_results[0],
                'accuracy': test_results[1]
            }
            print(f"   {name} - Loss: {test_results[0]:.4f}, Acc: {test_results[1]:.4f}")
        
        # Evaluate ensemble
        ensemble_results = self._evaluate_ensemble(test_dataset)
        
        # Plot results
        self._plot_results(individual_results, ensemble_results)
        
        # Save results
        self._save_results(individual_results, ensemble_results)
        
        return individual_results, ensemble_results
    
    def _evaluate_ensemble(self, test_dataset):
        """Evaluate ensemble"""
        print("Evaluating Ensemble...")
        
        # Calculate predictions
        frames_list = []
        masks_list = []
        for batch in test_dataset:
            f, m = batch
            frames_list.append(f.numpy())
            masks_list.append(m.numpy())
        
        frames = np.concatenate(frames_list, axis=0)
        true_masks = np.concatenate(masks_list, axis=0)
        
        # Predict with all models
        all_predictions = []
        for name, model in self.models.items():
            pred = model.predict(frames, verbose=0, batch_size=16)
            all_predictions.append(pred)
        
        # Combine predictions
        if self.meta_learner == 'averaging':
            avg_predictions = np.mean(all_predictions, axis=0)
            ensemble_pred = np.argmax(avg_predictions, axis=-1)
        else:
            # Use meta-learner
            features, true_labels = self.extract_features(test_dataset)
            if features.size > 0:
                ensemble_pred_labels = self.meta_learner.predict(features)
                # Convert labels to masks
                ensemble_pred = np.zeros_like(true_masks)
                for i, label in enumerate(ensemble_pred_labels):
                    ensemble_pred[i] = label
            else:
                avg_predictions = np.mean(all_predictions, axis=0)
                ensemble_pred = np.argmax(avg_predictions, axis=-1)
        
        # Calculate metrics
        accuracy = accuracy_score(true_masks.flatten(), ensemble_pred.flatten())
        
        # Calculate Dice for each class
        dice_scores = self.calculate_dice_scores(true_masks.flatten(), ensemble_pred.flatten())
        
        mean_dice = np.mean(list(dice_scores.values()))
        
        return {
            'accuracy': accuracy,
            'mean_dice': mean_dice,
            'dice_scores': dice_scores,
            'meta_learner': self.meta_learner_name
        }
    
    def _plot_results(self, individual_results, ensemble_results):
        """Plot result graphs"""
        # Accuracy plot
        plt.figure(figsize=(12, 6))
        
        model_names = list(individual_results.keys())
        accuracies = [individual_results[name]['accuracy'] for name in model_names]
        accuracies.append(ensemble_results['accuracy'])
        
        x_pos = np.arange(len(model_names) + 1)
        bars = plt.bar(x_pos, accuracies, color=['blue', 'green', 'red'])
        plt.ylabel('Accuracy')
        plt.title('Model Accuracies Comparison')
        plt.xticks(x_pos, model_names + ['Ensemble'])
        plt.ylim(0, 1)
        
        # Add numbers on plot
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Dice plot for each class
        class_names = ['Background', 'LV', 'Myocardium', 'RV']
        dice_values = [ensemble_results['dice_scores'][i] for i in range(4)]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, dice_values, color=['gray', 'red', 'green', 'blue'])
        plt.ylabel('Dice Coefficient')
        plt.title('Dice Scores per Class - Ensemble')
        plt.ylim(0, 1)
        
        for bar, dice in zip(bars, dice_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{dice:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'dice_per_class.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_results(self, individual_results, ensemble_results):
        """Save results"""
        results = {
            'individual_results': individual_results,
            'ensemble_results': ensemble_results,
            'timestamp': datetime.now().isoformat(),
            'model_dir': self.model_dir
        }
        
        with open(os.path.join(self.model_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {self.model_dir}")

# ---------------------- Main Execution ----------------------
def main():
    print("Starting Fixed Ensemble Cardiac Segmentation (150 Epochs)...")
    
    try:
        # 1. Load data loader
        from data_loaders.camus_hdf5_loader_fixed import CamusHDF5LoaderFixed
        data_loader = CamusHDF5LoaderFixed()
        print("Data loader loaded successfully")
        
        # 2. Split data
        total_samples = 450
        indices = list(range(total_samples))
        
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        # 3. Prepare data - using only one view and phase
        data_pipeline = AdvancedDataPipelineFixed(data_loader, target_size=(192, 192))
        
        print("Preparing training data...")
        train_dataset = data_pipeline.prepare_data(train_indices, view='2ch', phase='ED', batch_size=8, augment=True)
        
        print("Preparing validation data...")
        val_dataset = data_pipeline.prepare_data(val_indices, view='2ch', phase='ED', batch_size=8, augment=False)
        
        print("Preparing test data...")
        test_dataset = data_pipeline.prepare_data(test_indices, view='2ch', phase='ED', batch_size=8, augment=False)
        
        # 4. Build and train ensemble
        ensemble = EnsembleCardiacSegmentationFixed(input_shape=(192, 192, 1), num_classes=4)
        
        # Check saved models
        models_exist = ensemble.load_saved_models()
        
        if not models_exist:
            print("Training from scratch...")
            ensemble.compile_models(learning_rate=1e-3)
            
            print("="*60)
            print("Training Base Models for 150 Epochs...")
            print("="*60)
            
            histories = ensemble.train_models(train_dataset, val_dataset, epochs=150)
            print("Base models training completed")
        else:
            print("Using pre-trained models")
        
        # 5. Train meta-learner
        print("="*60)
        print("Training Meta-Learner...")
        print("="*60)
        
        meta_accuracy = ensemble.train_meta_learner(train_dataset, val_dataset)
        print(f"Meta-learner training completed (accuracy: {meta_accuracy:.4f})")
        
        # 6. Final evaluation
        print("="*60)
        print("Final Evaluation...")
        print("="*60)
        
        individual_results, ensemble_results = ensemble.evaluate_complete(test_dataset)
        
        # 7. Show final results
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"Ensemble Mean Dice: {ensemble_results['mean_dice']:.4f}")
        print(f"Meta-Learner: {ensemble_results['meta_learner']}")
        print(f"Results saved in: {ensemble.model_dir}")
        
        print("Dice Scores per Class:")
        class_names = ['Background', 'LV', 'Myocardium', 'RV']
        for i, class_name in enumerate(class_names):
            dice_score = ensemble_results['dice_scores'][i]
            print(f"   {class_name}: {dice_score:.4f}")
        
        # Final evaluation
        if ensemble_results['accuracy'] > 0.95:
            print("GOAL ACHIEVED! Accuracy > 95%!")
        elif ensemble_results['accuracy'] > 0.90:
            print("Excellent! Accuracy > 90%!")
        elif ensemble_results['accuracy'] > 0.85:
            print("Good result, ready for medical application")
        else:
            print("Needs improvement for medical use")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()