import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import EarlyStopping

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
x = Dense(512, activation='relu')(x)  # Dense layer with ReLU activation
x = Dense(52, activation='relu')(x)  # Additional Dense layer for feature extraction
x = BatchNormalization()(x)  # Normalize the activations

# Embedding layer to produce the desired embedding dimension
embedding_dim = 94
x = Dense(embedding_dim, activation='tanh', name='embedding_layer')(x)

# Output layer for the embedding
embedding_output = Dense(embedding_dim * 3, name='embedding_output', activation='tanh')(x)

# Create the final model by combining the pre-trained base with custom layers
vgg_model = Model(inputs=base_model.input, outputs=embedding_output)

# Compile the model with triplet loss
optimizer = SGD(learning_rate=0.01)
loss = tfa.losses.TripletSemiHardLoss()
vgg_model.compile(loss=loss, optimizer=optimizer)

# Define EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with augmented data and early stopping
history = vgg_model.fit(
    datagen.flow(x_train, y_train, batch_size=40),
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

#-----------------------------------------------------------------------------------
# Necessary step to generate embeddings for the ensemble technique
# Predict embeddings for the validation set using the trained VGG16 model
res_net_embeddings = vgg_model.predict(x_val)

#--------------------------------------------------------------------------------------

# After training the model, get the learned embeddings from the embedding layer
embedding_layer = vgg_model.get_layer('embedding_layer')
embedding_weights = embedding_layer.get_weights()[0]

# Print the learned embeddings
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

#---------------------------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors
from keras.metrics import Metric
from sklearn.metrics import precision_score

# Generate embeddings for all images in the test set
test_embeddings = vgg_model.predict(x_test)

# Calculate the mean Precision@K metric
def mean_precision_at_k(y_true, embeddings, k=5):
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(embeddings)

    precisions = []
    for i in range(len(y_true)):
        query_embedding = embeddings[i]
        _, indices = knn_model.kneighbors([query_embedding])
        correct_matches = np.sum(y_true[indices[0]] == y_true[i])
        precisions.append(correct_matches / k)
    return np.mean(precisions)

# Calculate mean Precision@5
mean_precision_at_5 = mean_precision_at_k(y_test, test_embeddings, k=5)

# Print the result in the desired format
print("Mean Precision@5:", mean_precision_at_5)

#------------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors
from keras.metrics import Metric
from sklearn.metrics import precision_score

# Generate embeddings for all images in the test set
train_embeddings = vgg_model.predict(x_train)

# Calculate the mean Precision@K metric
def mean_precision_at_k(y_true, embeddings, k=5):
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(embeddings)

    precisions = []
    for i in range(len(y_true)):
        query_embedding = embeddings[i]
        _, indices = knn_model.kneighbors([query_embedding])
        correct_matches = np.sum(y_true[indices[0]] == y_true[i])
        precisions.append(correct_matches / k)
    return np.mean(precisions)

# Calculate mean Precision@5
mean_precision_at_5 = mean_precision_at_k(y_train, train_embeddings, k=5)

# Print the result in the desired format
print("Mean Precision@5:", mean_precision_at_5)

#-----------------------------------------------------------------------------------

# GRAPH OF TRAINING AND VALIDATION LOSS

import matplotlib.pyplot as plt

# Extract the training and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Save the plot as a PNG file
plt.savefig('training_validation_loss02.png')

# Show the plot
plt.show()

#---------------------------------------------------------------------------------------

# EMBEDDING CLUSTERING GRAPH USING K-MEANS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Assuming you have 'test_embeddings' and 'cifar10_class_names' defined
num_clusters = 10
cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(test_embeddings)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(test_embeddings)

# Create a scatter plot to visualize clusters
plt.figure(figsize=(10, 7))

for cluster_id, class_name in enumerate(cifar10_class_names):
    cluster_points = reduced_embeddings[cluster_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=class_name)

plt.title('Clustering Visualization')
plt.legend()

# Save the figure before displaying it
plt.savefig('clustering_visualization.png')
plt.show()

#-----------------------------------------------------------------------------

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
visualize_similar_images([8, 27, 48, 32, 52], test_embeddings, x_test)

