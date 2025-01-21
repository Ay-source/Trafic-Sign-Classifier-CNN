# Data Augmentation

import tensorflow as tf
import numpy as np
import time
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# TODO: Fill this in based on where you saved the training and testing data
training_file = "./data/train.p"
validation_file = "./data/valid.p"
testing_file = "./data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = shuffle(train['features'], train['labels'])

# Convert images to float32
X_train = X_train.astype('float32')

print("Script Started...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Create an ImageDataGenerator with moderate augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the ImageDataGenerator on the training data
datagen.fit(X_train)

num_augmented_images = len(X_train)*10  # Specify the number of augmented images you want to generate
batch_size = 4  # Specify the batch size for the generator

# Create lists to store the augmented images and labels
augmented_images = []
augmented_labels = []

# Generate augmented images
i = 0
for _ in range(num_augmented_images // batch_size):
    augmented_batch = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=False)
    for images, labels in augmented_batch:
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        if len(augmented_images) >= num_augmented_images:
            break
    if len(augmented_images) >= num_augmented_images:
        break
    if i % 100 == 0:
        print("Working..")
    i += 1

# Convert lists to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Verify the shape of the augmented data
print(f'Augmented images shape: {augmented_images.shape}')
print(f'Augmented labels shape: {augmented_labels.shape}')

# Ensure images are in the correct range for visualization
augmented_images = np.clip(augmented_images, 0, 255).astype('uint8')

# Plot some augmented images
fig, ax = plt.subplots(1, 4, figsize=(20, 10))
for i in range(4):
    ax[i].imshow(augmented_images[i])
    ax[i].axis('off')
plt.show()

# Save the augmented data to a pickle file
with open('./data/aug.p', 'wb') as pfile:
    pickle.dump(
        {
            'features': augmented_images,
            'labels': augmented_labels,
        },
        pfile, pickle.HIGHEST_PROTOCOL)

print('Data cached in pickle file.')
