# Install TensorFlow and Addons (if not installed)
!pip install tensorflow==2.14.0 tensorflow-addons==0.23.0

from tensorflow.keras.optimizers import SGD
import tensorflow_addons as tfa
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Load the DenseNet121 model pre-trained on ImageNet (excluding the top classification layer)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add custom layers on top of the DenseNet121 base model
x = base_model.output
x = GlobalAveragePooling2D()(x)    # Global average pooling layer
x = Dense(512, activation='tanh')(x)  # Fully connected layer with 512 units
x = Dense(52, activation='tanh')(x)   # Fully connected layer with 52 units
x = BatchNormalization()(x)           # Batch normalization layer

# Define embedding layer
embedding_dim = 64
x = Dense(embedding_dim, activation='relu', name='embedding_layer')(x)

# Output embedding layer
embedding_output = Dense(embedding_dim * 3, activation='relu', name='embedding_output')(x)

# Create the final model with custom layers
dense_model = Model(inputs=base_model.input, outputs=embedding_output)

# Compile the model with SGD optimizer and TripletSemiHardLoss
optimizer = SGD(learning_rate=0.01)
loss = tfa.losses.TripletSemiHardLoss()
dense_model.compile(loss=loss, optimizer=optimizer)

# Define the EarlyStopping callback to stop training when validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with data augmentation and early stopping
history = dense_model.fit(datagen.flow(x_train, y_train, batch_size=40), epochs=50, 
                          validation_data=(x_val, y_val), callbacks=[early_stopping])
# ---------------------------------------------------------------------------------

# Necessary step to generate embeddings for the ensemble technique
# Predict embeddings for the validation set using the trained DenseNet model
dense_net_embeddings = dense_model.predict(x_val)
# 'dense_net_embeddings' will be used later in the ensemble technique

# -------------------------------------------------------------------------------

# EMBEDDING OF DENSENET121 MODEL ALSO SAVE IT
# After training the model, extract the learned embeddings from the embedding layer
embedding_layer = dense_model.get_layer('embedding_layer')  # Get the embedding layer
embedding_weights = embedding_layer.get_weights()[0]        # Extract the learned weights

# Print the shape and values of the learned embeddings
print("Learned Embeddings Shape:", embedding_weights.shape)
print("Learned Embeddings:")
print(embedding_weights)

# Save the learned embeddings to a CSV file using pandas
import pandas as pd

# Create a DataFrame to store the embedding weights
df = pd.DataFrame(embedding_weights)

# Save the DataFrame to 'embeddings.csv' without the index
df.to_csv('embeddings.csv', index=False)

print("Embeddings saved to 'embeddings.csv'.")

# ---------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score

# Generate embeddings for all images in the test set
test_embeddings = dense_model.predict(x_test)

# Function to calculate the mean Precision@K metric
def mean_precision_at_k(y_true, embeddings, k=5):
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')  # Initialize KNN model
    knn_model.fit(embeddings)  # Fit the model on embeddings

    precisions = []
    for i in range(len(y_true)):
        query_embedding = embeddings[i]  # Get the query embedding
        _, indices = knn_model.kneighbors([query_embedding])  # Find the nearest neighbors

        # Calculate precision for the current query
        correct_matches = np.sum(y_true[indices[0]] == y_true[i])
        precisions.append(correct_matches / k)  # Precision = correct matches / k

    return np.mean(precisions)  # Return the mean of precisions

# Calculate mean Precision@5
mean_precision_at_5 = mean_precision_at_k(y_test, test_embeddings, k=5)

# Print the result in the desired format
print("Mean Precision@5:", mean_precision_at_5)

# ------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Generate embeddings for all images in the training set
train_embeddings = dense_model.predict(x_train)

# Function to calculate the mean Precision@K metric
def mean_precision_at_k(y_true, embeddings, k=5):
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')  # Initialize KNN model
    knn_model.fit(embeddings)  # Fit the KNN model on the embeddings

    precisions = []
    for i in range(len(y_true)):
        query_embedding = embeddings[i]  # Get the query embedding
        _, indices = knn_model.kneighbors([query_embedding])  # Find the nearest neighbors

        # Calculate precision for the current query
        correct_matches = np.sum(y_true[indices[0]] == y_true[i])
        precisions.append(correct_matches / k)  # Precision = correct matches / k

    return np.mean(precisions)  # Return the mean precision across all queries

# Calculate mean Precision@5 on the training set
mean_precision_at_5 = mean_precision_at_k(y_train, train_embeddings, k=5)

# Print the result in a formatted way
print("Mean Precision@5 on training set:", mean_precision_at_5)

#--------------------------------------------------------

# GRAPH OF TRAINING AND VALIDATION LOSS
import matplotlib.pyplot as plt

# Extract the training and validation loss from the history object
train_loss = history.history['loss']          # Training loss values
val_loss = history.history['val_loss']        # Validation loss values

# Plot the training and validation loss
plt.plot(train_loss, label='Training Loss')   # Plot training loss
plt.plot(val_loss, label='Validation Loss')   # Plot validation loss
plt.xlabel('Epoch')                           # Set x-axis label
plt.ylabel('Loss')                            # Set y-axis label
plt.title('Training and Validation Loss')     # Set plot title
plt.legend()                                  # Show legend

# Save the plot as a PNG file
plt.savefig('training_validation_loss.png')   # Save the plot as 'training_validation_loss.png'
plt.show()                                    # Display the plot

# ------------------------------------------------------------------------------------------

# EMBEDDING CLUSTERING GRAPH USING K-MEANS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Number of clusters (CIFAR-10 has 10 classes)
num_clusters = 10

# Class names for CIFAR-10 dataset
cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                       "dog", "frog", "horse", "ship", "truck"]

# Apply t-SNE for dimensionality reduction on test embeddings
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)  # Use 2 components for visualization
reduced_embeddings = tsne.fit_transform(test_embeddings)  # Reduce dimensions

# Apply K-Means clustering on the original test embeddings
kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # K-Means clustering
cluster_labels = kmeans.fit_predict(test_embeddings)  # Cluster assignments

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(10, 7))

for cluster_id, class_name in enumerate(cifar10_class_names):
    # Select points corresponding to each cluster
    cluster_points = reduced_embeddings[cluster_labels == cluster_id]
    # Scatter plot of points in this cluster
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=class_name)

plt.title('Clustering Visualization with t-SNE')  # Plot title
plt.legend()  # Display legend with class names

# Save the plot as a PNG file
plt.savefig('clustering_visualization.png')  # Save the figure
plt.show()  # Display the plot

#------------------------------------------------------------------------------------------------------

# VIUALIZE SIMILAR IMAGES 
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def visualize_similar_images(query_indices, embeddings, x_test, n_neighbors=11):
    # Create NearestNeighbors model with cosine similarity
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn_model.fit(embeddings)

    # Set the size of the figure and the subplot layout
    plt.figure(figsize=(13, 5))
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)  # Adjust spacing

    for query_image_idx in query_indices:
        # Find similar images using NearestNeighbors with cosine similarity
        query_embedding = embeddings[query_image_idx].reshape(1, -1)
        distances, neighbor_indices = knn_model.kneighbors(query_embedding)

        # Visualize query image and similar images
        for i, idx in enumerate([query_image_idx] + list(neighbor_indices[0])):
            plt.subplot(1, 12, i + 1)  # Change the subplot to accommodate n_neighbors images

            # Resize the image for better visualization
            resized_image = resize(x_test[idx], (64, 64))  # Use x_test instead of x_train
            plt.imshow(resized_image)
            plt.axis('off')

    plt.show()

# Example usage:
visualize_similar_images([11, 27, 9, 14, 12], test_embeddings, x_test)

