from pipeline_onnx.utils import *
from pipeline_onnx.preprocess import *
from pipeline_onnx.postprocess import *
from PIL import Image, ImageDraw
from pipeline_onnx.configs import * #########

def prediction(session, image, cfg):
    image, ratio, (padd_left, padd_top) = resize_and_pad(image, new_shape=cfg.image_size)
    img_norm = normalization_input(image)
    pred = infer(session, img_norm)
    pred = postprocess(pred, cfg.conf_thres, cfg.iou_thres)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio
    return pred

def visualize(image, pred):
    img_ = image.copy()
    drawer = ImageDraw.Draw(img_)
    class_counts = {} # Create a dictionary to store class counts
    for p in pred:
        x1,y1,x2,y2,_, id = p
        id = int(id)
        drawer.rectangle((x1,y1,x2,y2),outline=IDX2COLORs[id],width=3)
        drawer.text((x1,y1), IDX2TAGs[id], fill="red") ##########

        class_name = IDX2TAGs[id]
        # Update the class count
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Iterate through the class counts and draw them on the image
    y_offset = 0  # Initialize the vertical offset
    for class_name, count in class_counts.items():
        text = f"{class_name}: {count}"
        drawer.text((10, y_offset), text, fill="green")  # You can customize the text color and position
        y_offset += 20

    return img_


