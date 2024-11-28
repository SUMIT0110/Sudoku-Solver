# train_model.py
from sudoku_utils import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def train_model():
    # Load MNIST dataset
    print("[INFO] loading MNIST dataset...")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # Add channel dimension and normalize pixel values
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1)) / 255.0
    testData = testData.reshape((testData.shape[0], 28, 28, 1)) / 255.0

    # Convert labels to one-hot encoding
    trainLabels = to_categorical(trainLabels, 10)
    testLabels = to_categorical(testLabels, 10)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        fill_mode="nearest"
    )

    # Build a deeper model
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    print("[INFO] training the model...")
    model.fit(datagen.flow(trainData, trainLabels, batch_size=64),
              validation_data=(testData, testLabels),
              epochs=50, verbose=1)

    # Save the model
    model.save("models/digit_classifier.h5")
    print("[INFO] model saved to 'models/digit_classifier.h5'")

    
    # Plot training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('models/training_history.png')
    plt.close()

if __name__ == "__main__":
    train_model()
