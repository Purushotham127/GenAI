import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('multi_class_image_classifier_1.h5')

# Class index mapping (should match training)
class_indices = {'cats': 0, 'dogs': 1, 'snakes': 2}
labels = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    predicted_label = labels[predicted_class]

    return predicted_label

# Example usage
img_path = 'test_cat.jpg'  # Replace with your image path
predicted_label = predict_image(img_path)
print(f"Predicted label: {predicted_label}")
