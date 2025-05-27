import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1: Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',              # Replace with your actual path
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'      # For multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    'dataset/validation',         # Replace with your actual path
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Reverse mapping for predictions
labels = {v: k for k, v in class_indices.items()}
print("Reverse mapping of class indices:", labels)

# Step 2: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer for multi-class
])

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Step 5: Visualize Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Step 6: Save the Model
model.save('multi_class_image_classifier_1.h5')

val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")