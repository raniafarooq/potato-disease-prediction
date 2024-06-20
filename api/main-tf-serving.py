# http://localhost:8000/predict ---paste in postman
# https://fastapi.tiangolo.com/tutorial/request-files/

# Run this docker command in windows powershell
# docker run -t --rm -p 8503:8503 -v "C:\Users\Rania Mehreen Farooq\Mini Project\MiniProject\Plant Leaf Disease\CodeBasicsProject:/CodeBasicsProject" tensorflow/serving --rest_api_port=8503 --model_config_file=/CodeBasicsProject/models.config
#docker ps
#docker stop <container id>
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

import tensorflow as tf
import keras
import requests

app = FastAPI()
endpoint= "http://localhost:8503/v1/models/potatoes_model:predict"

MODEL= tf.keras.models.load_model("saved_models/1/1.keras")

CLASS_NAMES=["Early Blight", "Late Blight", "Healthy"]

@app.get('/ping')
async def ping():
    return 'Hello i am alive!'

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image= read_file_as_image(await file.read())
    image_batch=np.expand_dims(image, axis=0)
    # predictions=MODEL.predict(image_batch)
    response=requests.post(endpoint, json={"instances": image_batch.tolist()})
    prediction=np.array(response.json()["predictions"][0])
    predicted_class= CLASS_NAMES[np.argmax(prediction)]
    confidence=np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)