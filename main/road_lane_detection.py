import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 #type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D , Dropout #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore

def load_and_preprocess_images_from_txt(txt_file_path, classes):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    
    directory = os.path.dirname(txt_file_path)
    
    images = []
    labels = []
    for line in lines:
        parts = line.strip().split()
        image_file = parts[0]
        image_path = os.path.join(directory, image_file)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        resized = cv2.resize(blurred, (224, 224))
        images.append(resized)
        
        labels.append(1 if len(parts) > 1 else 0)
    
    images = np.array(images)[..., np.newaxis]
    labels = np.array(labels)
    
    return images, labels

def load_classes(txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    classes = {line.strip(): i for i, line in enumerate(lines)}
    return classes

train_txt_path = r"D:\trial\final lane dataset.v2i.yolokeras\train\_annotations.txt"
valid_txt_path = r"D:\trial\final lane dataset.v2i.yolokeras\valid\_annotations.txt"
test_txt_path = r"D:\trial\final lane dataset.v2i.yolokeras\test\_annotations.txt"
train_classes_path = r"D:\trial\final lane dataset.v2i.yolokeras\train\_classes.txt"

classes = load_classes(train_classes_path)
print(f"Classes: {classes}")

X_train, y_train = load_and_preprocess_images_from_txt(train_txt_path, classes)
X_val, y_val = load_and_preprocess_images_from_txt(valid_txt_path, classes)
X_test, y_test = load_and_preprocess_images_from_txt(test_txt_path, classes)

print(f"Train data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

def lr_schedule(epoch):
    lr = 0.001
    if epoch > 30:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.5
    return lr

input_shape = (224, 224, 1)
base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)

batch_size = 8
epochs = 20

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
val_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')



# Save the entire model (architecture + weights + optimizer state)
model.save('road_detection_model.h5')



print("Model saved successfully.")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()