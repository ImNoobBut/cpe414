import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('uniform_model.h5')  # Replace with your model's path

# Define the input image size
INPUT_SIZE = 64

# Load and preprocess a single test image
def load_and_preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize pixel values
        return image
    return None

# Define a function to predict if the person is wearing a uniform and get confidence scores
def predict_uniform(image_path):
    test_image = load_and_preprocess_test_image(image_path)
    
    if test_image is not None:
        prediction = model.predict(test_image)
        class_index = np.argmax(prediction)
        confidence_score = prediction[0][class_index]
        
        classes = [ "Not Wearing Uniform","Wearing Uniform"]
        result = classes[class_index]
        
        return result, confidence_score
    else:
        return "Error loading the image.", None

# Test the model on a single image
image_path = r'D:\Document\4th year BSCPE\CPE414\Project\Uniform\Test\tn0001.png'  # Replace with the path to your test image
result, confidence = predict_uniform(image_path)

if confidence is not None:
    print(f'Test image: {image_path}')
    print(f'Prediction: {result}')
    print(f'Confidence Score: {confidence:.2f}')
else:
    print(f'Error loading the image.')

