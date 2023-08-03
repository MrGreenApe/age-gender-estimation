import base64
import requests
from PIL import Image
import io

# Open the image file in binary mode, convert it to base64, and decode to string
with open('D:\\Auditorium\\OpenCV\\python-gender-age-detect\\test.png', 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Send a POST request to the /predict endpoint with the base64 image in the JSON body
response = requests.post('http://localhost:8000/predict', json={'image': base64_image})

# Print the prediction from the response
print(response.json())