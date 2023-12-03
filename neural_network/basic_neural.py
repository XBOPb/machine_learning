import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load data. Data is 28x28 pixels greyscale value.
data = keras.datasets.fashion_mnist

# Split the data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Define labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shrink greyscale data down. 
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show an image
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()