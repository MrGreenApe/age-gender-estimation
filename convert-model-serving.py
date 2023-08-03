import tensorflow as tf
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from src.factory import get_model
from omegaconf import OmegaConf
from pathlib import Path
import os

model_version = "0001"
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

print(f"Model Input Tensor: {model.input_names[0]}")

print(f"Exporting Version: {model_version}")
export_path = f"./prod/{model_version}"

tf.saved_model.save(model, export_path)