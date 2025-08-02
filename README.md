
## 🎯 Objectives
- Generate robust image embeddings from CIFAR-10 using pre-trained CNNs.
- Improve embedding quality with fine-tuning and triplet loss.
- Evaluate retrieval performance using mean Precision@5 (mAP@5).
- Apply clustering (K-Means + t-SNE) for embedding visualization.
- Build a stacking ensemble model for combining multiple embeddings.
 ## 📂 Project Structure
 cifar10-image-embedding/
├── preprocessing.py # Dataset loading, normalization, splitting, augmentation
├── densenet121_model.py # DenseNet121-based embedding model
├── resnet50_model.py # ResNet50-based embedding model
├── vgg16_model.py # VGG16-based embedding model
├── stacking_ensemble.py # Combines embeddings via Stacking Ensemble Technique
├── requirements.txt # Required Python packages
├── README.md # Project description and instructions
├── embeddings.csv # Saved DenseNet embeddings
├── training_validation_loss.png # Loss graph image
└── clustering_visualization.png # Embedding cluster visualization

## 🖼️ Dataset
- **CIFAR-10**: 60,000 32x32 color images in 10 classes.
- Downloaded using: `keras.datasets.cifar10`

## 🧠 Models Used
- **DenseNet121** 
- **ResNet50**
- **VGG16**
- **Stacking Ensemble** 

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
