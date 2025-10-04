import keras
import numpy as np

from app.class_names import CLASS_NAMES

img_path = "./images/apple.png"
efficientnet_model = keras.models.load_model('efficientnet.h5')
# Load and preprocess the image
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
img_array = keras.applications.efficientnet.preprocess_input(img_array) # Preprocess for EfficientNet

# Make a prediction
predictions = efficientnet_model.predict(img_array)

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Get the class labels from the training data generator
# Assuming train_images is available from a previous cell
labels = CLASS_NAMES
labels = dict((v, k) for k, v in labels.items())

# Get the predicted class label
predicted_class_label = labels[predicted_class_index]

print(f"The predicted class for the image at {img_path} is: {predicted_class_label}")