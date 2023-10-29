# app = FastAPI()

# @app.get('/')
# async def home():
#     return "day la home"

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import cv2
from io import BytesIO
from pipeline_onnx.configs import *
from pipeline_onnx.utils import *
from pipeline_onnx.main import prediction, visualize
from fastapi.responses import StreamingResponse 
import io
app = FastAPI()

PATH_MODEL = "models/best.onnx"  # Update with the actual model path

session = load_session(PATH_MODEL)

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.05
    iou_thres = 0.3

cfg = CFG()

@app.post("/predict/")
async def predict_image(file: UploadFile):
    if file.content_type.startswith('image'):
        image = Image.open(BytesIO(await file.read()))
        image = image.convert("RGB")

        # Convert PIL image to a NumPy array
        image_np = np.array(image)

        # Make the prediction
        pred = prediction(session=session, image=image_np, cfg=cfg)
        
        # Convert the NumPy array back to a PIL image for visualization
        image = Image.fromarray(image_np)

        result_image = visualize(image, pred)

        # Convert the result image to binary data
        result_image_bytes = BytesIO()
        result_image.save(result_image_bytes, format="PNG")
        result_image_bytes.seek(0)

        # Return the image as binary data with the appropriate content type
        return StreamingResponse(io.BytesIO(result_image_bytes.read()), media_type="image/png")
    else:
        return {"error": "Invalid file format. Please upload an image."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
