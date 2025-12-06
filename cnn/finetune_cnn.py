import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import cv2
import os
from pathlib import Path
from tqdm import tqdm

alphabet_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
                   "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

print("Loading images from dataset...")
image_data = []
label_data = []

data_directory = Path('data/asl_alphabet_train')
image_size = 64

for letter in alphabet_labels:
    letter_folder = data_directory / letter
    if letter_folder.exists():
        image_files = list(letter_folder.glob("*.jpg")) + list(letter_folder.glob("*.JPG")) + \
                     list(letter_folder.glob("*.jpeg")) + list(letter_folder.glob("*.JPEG"))
        
        label_index = alphabet_labels.index(letter)
        
        print(f"Loading {len(image_files)} images for letter '{letter}'...")
        
        for img_path in tqdm(image_files, desc=f"  {letter}", leave=False):
            img = cv2.imread(str(img_path))
            if img is not None:
                img_resized = cv2.resize(img, (image_size, image_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                image_data.append(img_rgb)
                label_data.append(label_index)
    else:
        print(f"Warning: {letter_folder} not found")

X = np.array(image_data)
y = np.array(label_data)

print(f"\nLoaded {len(X)} images")
print(f"Image shape: {X.shape}")

X = X.astype('float32') / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

y_train_categorical = to_categorical(y_train, num_classes=26)
y_test_categorical = to_categorical(y_test, num_classes=26)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

def create_small_vgg(input_shape=(64, 64, 3), num_classes=26):
    model = Sequential(name='small_vgg')
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

print("\nCreating small VGG model...")
model = create_small_vgg(input_shape=(image_size, image_size, 3), num_classes=26)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nStarting training...")
history = model.fit(
    X_train,
    y_train_categorical,
    batch_size=32,
    epochs=20,
    validation_split=0.1,
    verbose=1
)

print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_categorical,
    verbose=1
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

model.save('vgg_asl_model.h5')
print("\nModel saved as 'vgg_asl_model.h5'")
