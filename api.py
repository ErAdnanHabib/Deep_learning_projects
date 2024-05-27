from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
import tensorflow as tf

app = FastAPI()

MODEL=tf.keras.models.load_model("saved_models/1.0")
CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive!" 




def read_file_as_image(data)-> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile= File(...)
):
    image=read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])*100
    return{
        'Predicted ' : predicted_class,
        'Confidence' : f"{float(confidence)} %",
    }
    




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
