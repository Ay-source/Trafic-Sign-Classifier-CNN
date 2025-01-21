# Data Augmentation

import tensorflow as tf
import numpy as np
import time

# Load pickled data
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./data/train.p"
validation_file="./data/valid.p"
testing_file = "./data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = shuffle(train['features'], train['labels'])




print("Script Started...")




from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)


num_augmented_images = 100  # Specify the number of augmented images you want to generate
batch_size = 4  # Specify the batch size for the generator

# Create an empty array to store the augmented images
augmented_images = []
augmented_labels = []

for _ in range(num_augmented_images // batch_size):
    augmented_batch = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=False)
    for images, labels in augmented_batch:
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        if len(augmented_images) >= num_augmented_images:
            break
    if len(augmented_images) >= num_augmented_images:
        break

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Verify the shape of the augmented data
print(f'Augmented images shape: {augmented_images.shape}')
print(f'Augmented labels shape: {augmented_labels.shape}')

# Plot some augmented images
"""
fig, ax = plt.subplots(1, 4, figsize=(20, 10))
for i in range(4):
    ax[i].imshow(augmented_images[i])
#plt.show()
"""

plt.imshow(augmented_images[10])
plt.show()



with open('./data/aug.p', 'wb') as pfile:
    pickle.dump(
        {
            'features': augmented_images,
            'labels': augmented_labels,
        },
        pfile, pickle.HIGHEST_PROTOCOL)

print('Data cached in pickle file.')