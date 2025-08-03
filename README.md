
### Objectives
```
-	To evaluate the performance of DenseNet 121, ResNet 50, and VGG 16 models for image embedding across diverse visual domains.
-	To Assess the effectiveness of K-nearest neighbors algorithm in measuring image similarity using embeddings.
-	To Examine the performance of MAP@5 metric in evaluating the performance of image embedding models.
-	Explore the enhancement of model performance through the utilization of Stacking Ensemble Technique in image embedding task.
-	Investigate the effects of fine-tuning and hyperparameter tuning on the performance of deep CNN models and stacking ensemble technique.
```
 ### Project Structure
```
cifar10-image-embedding/
│
├── preprocessing.py              # Loads CIFAR-10 dataset, applies normalization, splits & augmentations
│
├── densenet121_model.py          # Builds, trains, and extracts embeddings using DenseNet121
├── resnet50_model.py             # Builds, trains, and extracts embeddings using ResNet50
├── vgg16_model.py                # Builds, trains, and extracts embeddings using VGG16
│
├── stacking_ensemble.py          # Combines features from models using Stacking Ensemble Technique
│
├── embeddings.csv                # CSV file containing saved image embeddings
│
├── requirements.txt              # List of Python libraries needed to run the code
├── README.md                     # Project overview, instructions, results, and visualizations
│
└── assets/                       #  Store graphs and images
    ├── densenet_loss.png
    ├── resnet_loss.png
    ├── vgg_loss.png
    └── clustering_visual.png
```
### Dataset
- **CIFAR-10**: 60,000 32x32 color images.
- **Source**: `https://www.cs.toronto.edu/~kriz/cifar.html`

### Models Used
- DenseNet121 
- ResNet50
- VGG16
- Stacking Ensemble 

## 📊 Evaluation Metric
- **mAP@5 (mean Average Precision at 5)**

## ⚙️ Requirements
Install the necessary libraries with:

## 🚀 How to Run
# Step 1: Preprocess and augment CIFAR-10 data
python preprocessing.py

# Step 2: Train individual models
python densenet121_model.py
python resnet50_model.py
python vgg16_model.py

# Step 3: Combine embeddings using stacking
python stacking_ensemble.py

## 🧪 Results

| Model        | Training Accuracy | Test Accuracy | mAP@5 Score |
|--------------|-------------------|---------------|-------------|
| DenseNet121  | 85.48%            | 81.32%        | **83.71%**  |
| ResNet50     | 82.44%            | 79.15%        | -           |
| VGG16        | 85.54%            | 82.70%        | -           |
| Stacking     | 92.86%            | 87.45%        | -           |
