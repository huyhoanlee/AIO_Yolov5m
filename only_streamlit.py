import io
import streamlit as st
from PIL import Image
import torch
from pipeline_onnx.utils import *
from pipeline_onnx.configs import *
from pipeline_onnx.main import *
device = 'cpu'

# Load the pre-trained model ( chỉ một lần)
@st.cache_resource()
def load_model():
    print("Here")
    model = load_session(
        path = 'models/best.onnx',
    )
    return model 

# Load model
session = load_model()

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.05
    iou_thres = 0.3

cfg = CFG()

# Streamlit UI
st.title("Image Classification with Streamlit")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform inference with the model
    # image = Image.open(uploaded_image)
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    #print(image)

    pred = prediction(
        session=session,
        image=image,
        cfg=cfg
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    predicted_image = visualize(image, pred)

    st.image(predicted_image, caption="Result Image", use_column_width=True)
