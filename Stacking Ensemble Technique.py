import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

import numpy as np

# Subsample dense_net_embeddings to match the size of res_net_embeddings and vgg_embeddings
dense_net_embeddings_subsampled = dense_net_embeddings[:5000]

# Stack the embeddings
stacked_embeddings = np.hstack((dense_net_embeddings_subsampled, res_net_embeddings, vgg_embeddings))


# Create a meta-model for embedding (output dimension should match the combined embeddings dimension)
embedding_dim = stacked_embeddings.shape[1]
meta_model = Sequential()
meta_model.add(Dense(embedding_dim, activation='LeakyReLU', input_shape=(stacked_embeddings.shape[1],)))


# Compile the meta-model
#meta_model.compile(loss='mean_squared_error', optimizer=Adam())

meta_model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.1))

from keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the meta-model with early stopping
meta_model.fit(stacked_embeddings, stacked_embeddings, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Use the trained meta-model for predictions on the test set
test_dense_net_embeddings = dense_model.predict(x_test)
test_res_net_embeddings = res_model.predict(x_test)
test_vgg_embeddings = vgg_model.predict(x_test)

test_stacked_embeddings = np.hstack((test_dense_net_embeddings, test_res_net_embeddings, test_vgg_embeddings))
test_predictions = meta_model.predict(test_stacked_embeddings)

#------------------------------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors
# Generate embeddings for all images in the test set
test_embeddings = meta_model.predict(test_stacked_embeddings)  # Use meta_model for prediction

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

# Calculate Mean Precision@5
mean_precision_at_5 = mean_precision_at_k(y_test, test_embeddings, k=5)

# Print the result
print("Mean Precision@5:", mean_precision_at_5)

#-----------------------------------------------------------------------------

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
plt.savefig('ensemble_training_validation_loss.png')

# Show the plot
plt.show()

#--------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------