#This file will handle the dataset loading, normalization, splitting, and augmentation.

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Display the shape of the training and testing datasets
print('Training data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)

# Count and display the number of unique classes
num_classes = len(set(y_train.flatten()))
print('Number of classes:', num_classes)

#DATASET SPLITTING
# Split the data into 80% train/validation and 20% test
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)
# From the 80% (train/validation), split into 70% training and 10% validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.125, random_state=42)  # 0.125 of 80% gives 10%
# Now 70% train, 10% validation, and 20% test

# Display the shape of the datasets after splitting
print('Training data shape:', x_train.shape)
print('Validation data shape:', x_val.shape)
print('Testing data shape:', x_test.shape)


# DATA AUGMENTATION
# Choose and display an image from the CIFAR-10 dataset
image_index = 9
img = x_train[image_index]
plt.figure(figsize=(1, 1))
plt.imshow(img)
print('Original Image:')
plt.show()
# Reshape the image for augmentation
x = img.reshape((1,) + img.shape)

# Configure data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,         # Randomly rotate images by up to 20 degrees
    width_shift_range=0.1,     # Horizontally shift images by up to 10% of width
    height_shift_range=0.1,    # Vertically shift images by up to 10% of height
    horizontal_flip=True       # Randomly flip images horizontally
)

# Display augmented images
print('Augmented Images:')
plt.figure(figsize=(6, 6))
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='augmented_images', save_prefix='aug', save_format='jpeg'):
    plt.subplot(5, 5, i + 1)
    plt.imshow(batch[0])  # Display the augmented image
    i += 1
    if i % 10 == 0:       # Stop after generating 10 augmented images
        break

# Save the grid of augmented images
plt.tight_layout()
plt.savefig('augmented_images/augmented_images.png')
plt.show()
print("Augmented images are saved in the 'augmented_images' directory.")