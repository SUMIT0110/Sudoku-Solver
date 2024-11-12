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

def train_model():
    # Initialize parameters
    INIT_LR = 1e-3
    EPOCHS = 50
    BS = 64
    
    # Load MNIST dataset
    print("[INFO] accessing MNIST...")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    
    # Preprocess the data
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))
    trainData = trainData.astype("float32") / 255.0
    testData = testData.astype("float32") / 255.0
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Convert labels
    le = LabelBinarizer()
    trainLabels = le.fit_transform(trainLabels)
    testLabels = le.transform(testLabels)
    
    # Initialize and compile model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # Learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train network
    print("[INFO] training network...")
    H = model.fit(
        datagen.flow(trainData, trainLabels, batch_size=BS),
        validation_data=(testData, testLabels),
        steps_per_epoch=len(trainData) // BS,
        epochs=EPOCHS,
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Evaluate network
    print("[INFO] evaluating network...")
    predictions = model.predict(testData)
    print(classification_report(
        testLabels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in le.classes_]))
    
    # Save model
    model.save('models/digit_classifier.h5')
    print("[INFO] Model saved as 'models/digit_classifier.h5'")
    
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