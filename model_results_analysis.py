# model_results_analysis.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
import cv2
from datetime import datetime

class ModelResultsAnalyzer:
    def __init__(self, model_dir, data_loader, target_size=(192, 192)):
        self.model_dir = model_dir
        self.data_loader = data_loader
        self.target_size = target_size
        self.models = {}
        self.num_classes = 4
        self.class_names = ['Background', 'LV', 'Myocardium', 'RV']
        
    def load_models(self):
        """Load all saved models"""
        print("Loading models...")
        
        model_files = {
            'unet_advanced': 'best_unet_advanced.h5',
            'deep_unet_improved': 'best_deep_unet_improved.h5'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                try:
                    self.models[name] = load_model(model_path)
                    print(f"Loaded {name}")
                except Exception as e:
                    print(f"Error loading {name}: {e}")
            else:
                print(f"Model file not found: {model_path}")
        
        return len(self.models) > 0
    
    def prepare_test_data(self, test_indices, view='2ch', phase='ED'):
        """Prepare test data"""
        print("Preparing test data...")
        
        frames = []
        masks = []
        
        for idx in test_indices:
            frame_key = f"{view}_{phase}_frame"
            mask_key = f"{view}_{phase}_mask"
            
            sample_data = self.data_loader.load_patient_data(idx)
            frame = sample_data.get(frame_key)
            mask = sample_data.get(mask_key)
            
            if frame is not None and mask is not None:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                
                frames.append(frame)
                masks.append(mask)
        
        frames = np.array(frames)[..., np.newaxis]
        masks = np.array(masks)
        
        print(f"Test data prepared: {frames.shape} frames, {masks.shape} masks")
        return frames, masks
    
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
    
    def flatten_predictions_for_cm(self, y_true, y_pred):
        """Flatten predictions for confusion matrix"""
        # If y_pred shape is like y_true (height, width)
        if y_pred.shape == y_true.shape:
            return y_true.flatten(), y_pred.flatten()
        
        # If y_pred has shape (samples, height, width, classes)
        elif y_pred.ndim == 4:
            y_pred_classes = np.argmax(y_pred, axis=-1)
            return y_true.flatten(), y_pred_classes.flatten()
        
        else:
            print(f"Unexpected shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
            return y_true.flatten(), y_pred.flatten()
    
    def plot_confusion_matrix_all_models(self, true_masks, predictions_class, save_path=None):
        """Plot confusion matrix for all models - fixed"""
        n_models = len(predictions_class)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 5))
        
        if n_models == 1:
            axes = [axes]
        
        # Matrix for each model
        for idx, (model_name, y_pred) in enumerate(predictions_class.items()):
            # Flatten predictions
            y_true_flat, y_pred_flat = self.flatten_predictions_for_cm(true_masks, y_pred)
            
            cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(self.num_classes))
            
            # Normalization
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
            
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'Confusion Matrix\n{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        # Ensemble matrix (average)
        if n_models > 1:
            # Average probabilities for ensemble
            ensemble_probs = []
            for model_name, model in self.models.items():
                ensemble_probs.append(model.predict(true_masks[..., np.newaxis], verbose=0, batch_size=16))
            
            ensemble_prob = np.mean(ensemble_probs, axis=0)
            ensemble_class = np.argmax(ensemble_prob, axis=-1)
            
            y_true_flat, ensemble_flat = self.flatten_predictions_for_cm(true_masks, ensemble_class)
            
            cm_ensemble = confusion_matrix(y_true_flat, ensemble_flat, labels=range(self.num_classes))
            cm_ensemble_normalized = cm_ensemble.astype('float') / (cm_ensemble.sum(axis=1)[:, np.newaxis] + 1e-8)
            
            sns.heatmap(cm_ensemble_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=axes[-1], cbar=True)
            axes[-1].set_title('Confusion Matrix\nEnsemble')
            axes[-1].set_xlabel('Predicted')
            axes[-1].set_ylabel('True')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved: {save_path}")
        plt.show()
    
    def plot_roc_curves_all_models(self, true_masks, predictions_prob, save_path=None):
        """Plot ROC curves for all models and classes - fixed"""
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        linestyles = ['-', '--', '-.', ':']
        
        # For each model
        for model_idx, (model_name, y_pred_prob) in enumerate(predictions_prob.items()):
            # Flatten data
            if y_pred_prob.ndim == 4:  # shape (samples, height, width, classes)
                y_pred_flat = y_pred_prob.reshape(-1, self.num_classes)
                y_true_flat = true_masks.flatten()
            else:
                print(f"Skipping ROC for {model_name} - unexpected shape: {y_pred_prob.shape}")
                continue
            
            # Binarize the labels
            y_true_bin = label_binarize(y_true_flat, classes=range(self.num_classes))
            
            # For each class
            for class_id in range(self.num_classes):
                try:
                    # Calculate ROC
                    fpr, tpr, _ = roc_curve(y_true_bin[:, class_id], y_pred_flat[:, class_id])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, 
                            color=colors[model_idx % len(colors)],
                            linestyle=linestyles[class_id % len(linestyles)],
                            label=f'{model_name} - {self.class_names[class_id]} (AUC = {roc_auc:.3f})',
                            linewidth=2)
                except Exception as e:
                    print(f"Error plotting ROC for {model_name} class {class_id}: {e}")
                    continue
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models & Classes')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved: {save_path}")
        plt.show()
    
    def plot_dice_comparison(self, dice_results_dict, save_path=None):
        """Plot Dice scores comparison for all models and classes"""
        models = list(dice_results_dict.keys())
        class_names = self.class_names
        
        # Prepare data
        dice_data = []
        for model_name, dice_scores in dice_results_dict.items():
            for class_id, class_name in enumerate(class_names):
                dice_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Dice Score': dice_scores[class_id]
                })
        
        df = pd.DataFrame(dice_data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=df, x='Class', y='Dice Score', hue='Model', palette='viridis')
        
        # Add numbers on plot
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.title('Dice Scores Comparison - All Models & Classes', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Dice Coefficient', fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dice comparison saved: {save_path}")
        plt.show()
        
        return df
    
    def plot_sample_predictions(self, frames, true_masks, predictions_class, save_path=None):
        """Plot sample predictions"""
        n_samples = min(4, len(frames))
        n_models = len(predictions_class)
        
        fig, axes = plt.subplots(n_samples, n_models + 2, figsize=(4 * (n_models + 2), 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Original image
            axes[i, 0].imshow(frames[i, :, :, 0], cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # True mask
            axes[i, 1].imshow(true_masks[i], cmap='jet', vmin=0, vmax=3)
            axes[i, 1].set_title('True Mask')
            axes[i, 1].axis('off')
            
            # Prediction for each model
            for j, (model_name, y_pred) in enumerate(predictions_class.items()):
                axes[i, j + 2].imshow(y_pred[i], cmap='jet', vmin=0, vmax=3)
                axes[i, j + 2].set_title(f'{model_name}\nPrediction')
                axes[i, j + 2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved: {save_path}")
        plt.show()
    
    def generate_comprehensive_report(self, test_indices, view='2ch', phase='ED'):
        """Generate comprehensive report for all models"""
        print("Generating Comprehensive Analysis Report...")
        
        # Prepare test data
        frames, true_masks = self.prepare_test_data(test_indices, view, phase)
        
        # Predict with all models
        predictions_prob = {}
        predictions_class = {}
        dice_results = {}
        
        for model_name, model in self.models.items():
            print(f"Predicting with {model_name}...")
            
            # Probability predictions
            y_pred_prob = model.predict(frames, verbose=0, batch_size=16)
            predictions_prob[model_name] = y_pred_prob
            
            # Class predictions
            y_pred_class = np.argmax(y_pred_prob, axis=-1)
            predictions_class[model_name] = y_pred_class
            
            # Calculate Dice scores
            dice_scores = self.calculate_dice_scores(true_masks.flatten(), y_pred_class.flatten())
            dice_results[model_name] = dice_scores
            
            print(f"{model_name} - Dice Scores: {dice_scores}")
        
        # Ensemble predictions
        if len(self.models) > 1:
            ensemble_probs = []
            for model_name, model in self.models.items():
                ensemble_probs.append(model.predict(frames, verbose=0, batch_size=16))
            
            ensemble_prob = np.mean(ensemble_probs, axis=0)
            ensemble_class = np.argmax(ensemble_prob, axis=-1)
            predictions_prob['Ensemble'] = ensemble_prob
            predictions_class['Ensemble'] = ensemble_class
            
            dice_ensemble = self.calculate_dice_scores(true_masks.flatten(), ensemble_class.flatten())
            dice_results['Ensemble'] = dice_ensemble
            print(f"Ensemble - Dice Scores: {dice_ensemble}")
        
        # Create directory for saving results
        results_dir = os.path.join(self.model_dir, 'comprehensive_analysis')
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Plot confusion matrices
        print("Plotting confusion matrices...")
        cm_path = os.path.join(results_dir, 'confusion_matrices.png')
        self.plot_confusion_matrix_all_models(true_masks, predictions_class, cm_path)
        
        # 2. Plot ROC curves
        print("Plotting ROC curves...")
        roc_path = os.path.join(results_dir, 'roc_curves.png')
        self.plot_roc_curves_all_models(true_masks, predictions_prob, roc_path)
        
        # 3. Plot Dice scores comparison
        print("Plotting Dice scores comparison...")
        dice_path = os.path.join(results_dir, 'dice_comparison.png')
        dice_df = self.plot_dice_comparison(dice_results, dice_path)
        
        # 4. Plot sample predictions
        print("Plotting sample predictions...")
        sample_path = os.path.join(results_dir, 'sample_predictions.png')
        self.plot_sample_predictions(frames, true_masks, predictions_class, sample_path)
        
        # 5. Save numerical results
        print("Saving numerical results...")
        self.save_numerical_results(dice_results, predictions_class, true_masks, results_dir)
        
        # 6. Generate summary report
        self.generate_summary_report(dice_results, results_dir)
        
        print(f"Comprehensive analysis completed!")
        print(f"Results saved in: {results_dir}")
        
        return dice_results, dice_df
    
    def save_numerical_results(self, dice_results, predictions, true_masks, results_dir):
        """Save numerical results"""
        from sklearn.metrics import classification_report, accuracy_score
        
        results = {}
        
        for model_name, y_pred in predictions.items():
            # Flatten for metric calculation
            y_true_flat, y_pred_flat = self.flatten_predictions_for_cm(true_masks, y_pred)
            
            # Classification report
            report = classification_report(
                y_true_flat, 
                y_pred_flat,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Accuracy
            accuracy = accuracy_score(y_true_flat, y_pred_flat)
            
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'dice_scores': dice_results[model_name],
                'mean_dice': np.mean(list(dice_results[model_name].values()))
            }
        
        # Save as JSON
        with open(os.path.join(results_dir, 'numerical_results.json'), 'w') as f:
            # Convert numpy types to Python native types
            def convert_types(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_types(results), f, indent=4)
        
        # Save as CSV
        dice_data = []
        for model_name, dice_scores in dice_results.items():
            for class_id, class_name in enumerate(self.class_names):
                dice_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Dice_Score': dice_scores[class_id]
                })
        
        dice_df = pd.DataFrame(dice_data)
        dice_df.to_csv(os.path.join(results_dir, 'dice_scores.csv'), index=False)
        
        print("Numerical results saved")
    
    def generate_summary_report(self, dice_results, results_dir):
        """Generate summary report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COMPREHENSIVE MODEL ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model directory: {self.model_dir}")
        report_lines.append(f"Number of classes: {self.num_classes}")
        report_lines.append("")
        
        # Best model based on mean Dice
        best_model = None
        best_mean_dice = 0
        
        for model_name, dice_scores in dice_results.items():
            mean_dice = np.mean(list(dice_scores.values()))
            report_lines.append(f"{model_name.upper()}:")
            report_lines.append(f"   Mean Dice: {mean_dice:.4f}")
            
            for class_id, class_name in enumerate(self.class_names):
                report_lines.append(f"   {class_name}: {dice_scores[class_id]:.4f}")
            
            report_lines.append("")
            
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_model = model_name
        
        report_lines.append("BEST PERFORMING MODEL:")
        report_lines.append(f"   {best_model} with mean Dice: {best_mean_dice:.4f}")
        report_lines.append("")
        
        # Overall evaluation
        if best_mean_dice > 0.90:
            report_lines.append("EXCELLENT PERFORMANCE - Ready for clinical use!")
        elif best_mean_dice > 0.85:
            report_lines.append("GOOD PERFORMANCE - Suitable for medical application")
        elif best_mean_dice > 0.80:
            report_lines.append("ACCEPTABLE PERFORMANCE - May need minor improvements")
        else:
            report_lines.append("NEEDS IMPROVEMENT - Not ready for medical use")
        
        report_lines.append("=" * 60)
        
        # Save report
        report_path = os.path.join(results_dir, 'summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print report to console
        print('\n'.join(report_lines))
        
        print(f"Summary report saved: {report_path}")

# ---------------------- Main Execution ----------------------
def main():
    print("Starting Comprehensive Model Analysis...")
    
    try:
        # 1. Load data loader
        from data_loaders.camus_hdf5_loader_fixed import CamusHDF5LoaderFixed
        data_loader = CamusHDF5LoaderFixed()
        print("Data loader loaded successfully")
        
        # 2. Set model path
        model_dir = "C:\\Alex The Great\\Project\\ensemble_cardiac_fixed_20251005_143351"
        
        # 3. Create analyzer
        analyzer = ModelResultsAnalyzer(model_dir, data_loader)
        
        # 4. Load models
        if not analyzer.load_models():
            print("No models loaded. Exiting.")
            return
        
        # 5. Test data split (similar to main)
        total_samples = 450
        indices = list(range(total_samples))
        
        from sklearn.model_selection import train_test_split
        train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        print(f"Using {len(test_indices)} test samples")
        
        # 6. Generate comprehensive report
        dice_results, dice_df = analyzer.generate_comprehensive_report(test_indices)
        
        print("\nAnalysis completed successfully!")
        print(f"Results for {len(analyzer.models)} models")
        print(f"Best model performance:")
        
        best_model = None
        best_mean_dice = 0
        for model_name, scores in dice_results.items():
            mean_dice = np.mean(list(scores.values()))
            print(f"   {model_name}: {mean_dice:.4f}")
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_model = model_name
        
        print(f"Best: {best_model} ({best_mean_dice:.4f})")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()