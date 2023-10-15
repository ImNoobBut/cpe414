import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split  

# Set random seed for reproducibility
np.random.seed(42)

# Directory paths
image_directory = r'D:\Document\4th year BSCPE\CPE414\Project\Uniform'
uniform_directory = os.path.join(image_directory, 'uniform_students')
no_uniform_directory = os.path.join(image_directory, 'Not_uniform')

# Hyperparameters
INPUT_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the dataset
def load_and_preprocess_data(directory, label):
    dataset = []
    labels = []

    image_files = os.listdir(directory)
    for image_name in image_files:
        if image_name.endswith('.jpg'):
            try:
                image = cv2.imread(os.path.join(directory, image_name))
                if image is not None:
                    image = Image.fromarray(image, 'RGB')
                    image = image.resize((INPUT_SIZE, INPUT_SIZE))
                    dataset.append(np.array(image))
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_name}: {str(e)}")

    return np.array(dataset), np.array(labels)

data_uniform, labels_uniform = load_and_preprocess_data(uniform_directory, 1)
data_no_uniform, labels_no_uniform = load_and_preprocess_data(no_uniform_directory, 0)

# Combine data from both classes
X = np.concatenate((data_uniform, data_no_uniform), axis=0)
y = np.concatenate((labels_uniform, labels_no_uniform), axis=0)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0

# Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 output units for "uniform" and "no_uniform"

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['categorical_accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('uniform_best.h5', save_best_only=True)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Data augmentation during training
train_datagen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Train the model
history = model.fit(train_datagen, epochs=EPOCHS, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Save the final model
model.save('uniform_model.h5')
