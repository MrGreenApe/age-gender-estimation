# Model Serving
Instructions to serve the model using TensorFlow Serving
[https://www.tensorflow.org/tfx/guide/serving](https://www.tensorflow.org/tfx/guide/serving)

## Model Conversion
Before Serving the model needs to be converted from hdf5 to a protobuf format

```sh
# Activate the environment
conda activate age-gender-estimation

# Change the model_version="0001" in the file .\convert-model-serving.py

# Convert the model
python .\convert-model-serving.py
```

This will generate the folowing folder structure for the model:
```
age-gender-estimation
├── prod
│   ├── [model_version]
│   │   ├── assets
│   │   ├── variables
│   │   └── saved_model.pb
```

The tensorflow/serving docker container needs to be started using the path to the prod folder. 
TensorFlow Serving will automatically pickup the latest version and will also make a switch if a new version is generated while running.

```
docker run -t --rm -p 8501:8501 -v "[path_to_prod_folder]:/models/auditorium" -e MODEL_NAME=auditorium tensorflow/serving
```