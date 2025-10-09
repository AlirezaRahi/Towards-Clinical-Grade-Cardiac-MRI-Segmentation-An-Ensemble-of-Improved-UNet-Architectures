# Towards Clinical-Grade Cardiac MRI Segmentation: An Ensemble of Improved UNet Architectures

This repository contains the implementation of our research on cardiac segmentation using a hybrid deep learning ensemble model on echocardiography datasets, as presented in our comprehensive framework.

## 📄 Publication
**Title:** Towards Clinical-Grade Cardiac MRI Segmentation: An Ensemble of Improved UNet Architectures 
**Journal:** medRxiv  
**DOI:** https://doi.org/10.1101/2025.10.08.25337578 
**Publication Date:**  October 09, 2025.  
**URL:** https://www.medrxiv.org/content/10.1101/2025.10.08.25337578v1

## 📊 Abstract
Cardiac segmentation from echocardiography is crucial for diagnosing cardiovascular diseases and assessing cardiac function. In this study, we propose a hybrid deep learning ensemble model for automatic segmentation of cardiac structures from 2-chamber echocardiography views. Our framework integrates multiple U-Net architectures with advanced meta-learning techniques to combine their predictive strengths. Experimental evaluations on benchmark cardiac datasets demonstrate that the proposed ensemble achieves superior segmentation performance with high Dice coefficients across all cardiac structures. These results highlight the potential of hybrid ensemble methods as reliable computer-aided diagnostic tools for cardiology, contributing to accurate and efficient cardiac analysis.

## 🏗️ Model Architectures

### Advanced U-Net Model
A sophisticated U-Net architecture with increased capacity featuring:
- Enhanced encoder-decoder structure with residual connections
- Batch normalization and dropout for regularization
- 64-1024 filter progression with skip connections
- Advanced data augmentation pipeline

### Deep U-Net Improved
A deeper U-Net variant with:
- Expanded network depth for complex feature learning
- Dense convolutional blocks with he_normal initialization
- Multi-scale feature extraction capabilities
- Optimized gradient flow

### Meta-Learner Ensemble
The predictions from individual U-Net models are combined using:
- Feature extraction from model predictions (probabilities, entropy, confidence)
- Random Forest and Gradient Boosting meta-learners
- Optimal combination learning for final segmentation

## 📁 Dataset
The model was trained and evaluated on cardiac echocardiography data:

**Source:** CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)  
**Content:** 2-chamber view echocardiography images with expert annotations  
**Structures:** Background, Left Ventricle (LV), Myocardium, Right Ventricle (RV)  
**Phases:** End-diastole (ED) and End-systole (ES) frames

## 🚀 Installation & Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt

Key libraries include:

tensorflow>=2.8.0

keras>=2.8.0

scikit-learn>=1.0.2

opencv-python>=4.5.5

matplotlib>=3.5.1

seaborn>=0.11.2

numpy>=1.21.5

pandas>=1.4.2

joblib>=1.1.0

💻 Usage
Training the Ensemble Model
bash
python cardiac_ensemble_fixed_final.py
Comprehensive Model Analysis
bash
python model_results_analysis.py
Typical Workflow:
Data Preprocessing: Automatic resizing, normalization, and augmentation

Base Model Training: Individual U-Net models trained for 150 epochs

Meta-Learner Training: Ensemble combination optimization

Evaluation: Comprehensive performance analysis with multiple metrics

📊 Performance Metrics
The framework evaluates performance using:

Dice Coefficient: For each cardiac structure (Background, LV, Myocardium, RV)

Accuracy: Overall pixel-wise classification accuracy

Confusion Matrix: Per-class performance analysis

ROC Curves: Model discrimination capability

Visual Comparisons: Sample prediction visualization

 Project Structure
text
cardiac-segmentation-ensemble/
├── cardiac_ensemble_fixed_final.py    # Main ensemble training script
├── model_results_analysis.py          # Comprehensive analysis toolkit
├── ensemble_cardiac_fixed_*/          # Model directories (auto-generated)
│   ├── best_unet_advanced.h5
│   ├── best_deep_unet_improved.h5
└── requirements.txt

 Key Features
Advanced Data Pipeline: Intelligent preprocessing with class-aware augmentation

Multi-Model Ensemble: Combines strengths of different U-Net architectures

Meta-Learning: Learns optimal combination of base model predictions

Comprehensive Evaluation: Extensive metrics and visualization tools

Medical-Grade Validation: Ready for clinical application with performance thresholds

 Model Performance
Typical performance metrics achieved:

Ensemble Mean Dice: >0.85

LV Segmentation Dice: >0.90

Myocardium Dice: >0.80

Background Dice: >0.95

Overall Accuracy: >0.90

 Model Availability
Due to the large size of trained model files and dataset licensing considerations, the actual model weights and dataset are not publicly hosted in this repository.

The complete source code for data preprocessing, model architecture definition, training, and evaluation is provided, enabling full reproducibility of our results.

For academic collaboration or research verification requests requiring access to the pre-trained models, please contact the authors directly.

## Citation
If you use this work in your research, please cite our paper:

Rahi, A. (2025). Towards Clinical-Grade Cardiac MRI Segmentation: An Ensemble of Improved UNet Architectures. medRxiv. https://doi.org/10.1101/2025.10.08.25337578

If you use the code implementation in your work, please also cite:

Rahi, A. (2025). Towards Clinical-Grade Cardiac MRI Segmentation: An Ensemble of Improved UNet Architectures [Computer software]. GitHub repository, AlirezaRahi/Towards-Clinical-Grade-Cardiac-MRI-Segmentation-An-Ensemble-of-Improved-UNet-Architectures. Retrieved from https://github.com/AlirezaRahi/Towards-Clinical-Grade-Cardiac-MRI-Segmentation-An-Ensemble-of-Improved-UNet-Architectures


## Author
Alireza Rahi
 Email: alireza.rahi@outlook.com
 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/
 GitHub: https://github.com/AlirezaRahi


## License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

This means you are free to:

- **Share** — copy and redistribute the material in any medium or format for non-commercial purposes.

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

**Summary**: This work may be read and downloaded for personal use only. It may be shared in its complete and unaltered form for non-commercial purposes, provided that the author's name, the title of the work, and a link to the original source (this repository) and the license are clearly cited. Any modification, adaptation, commercial use, or distribution for profit is strictly prohibited.

For permissions beyond the scope of this license, please contact the author directly.

![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)

Copyright (c) 2025 Alireza Rahi
For licensing inquiries, please contact the author.

