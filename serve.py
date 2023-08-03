from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
import base64
import io
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model

app = Flask(__name__)

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))

 # load model and weights
model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
model = get_model(cfg)
model.load_weights(weight_file)

@app.route('/')
def index_view():
    return "Welcome to the age and gender prediction app!"


# def convertImage(imgData1):
#     imgstr = re.search(b'base64,(.*)',imgData1).group(1)
#     with open('output.png','wb') as output:
#         output.write(base64.b64decode(imgstr))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_data = base64.b64decode(data['image'])

    # Decode the base64 image    
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB and resize the image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))

    # Convert to numpy array and reshape
    image = np.array(image)
    image = image.reshape((1, img_size, img_size, 3))

    faces = np.empty((1, img_size, img_size, 3))
    faces[0] = image

    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()

    print(predicted_genders[0].shape)
    print(predicted_ages[0].shape)

    #return jsonify({'gender': predicted_genders, 'age': predicted_ages})
    return jsonify({'gender': (1 if predicted_genders[0][0]>0.5 else 0), 'age': int(predicted_ages[0])})
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)