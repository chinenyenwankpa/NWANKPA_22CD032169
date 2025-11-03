# model_training.py
# Optional training script. Run in an environment with GPU if you intend to train.
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Example params - adjust as needed
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7  # typical FER2013

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training flow depends on how you prepare your dataset/Generator.
# This file is illustrative only — use your Colab script for actual training.
if __name__ == "__main__":
    model = build_model(input_shape=(48,48,1), num_classes=NUM_CLASSES)
    model.summary()
    # Save as example
    model.save("face_emotionModel_example.h5")
    print("Saved example model as face_emotionModel_example.h5")
