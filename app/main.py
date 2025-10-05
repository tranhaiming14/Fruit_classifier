from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
import numpy as np
from app.class_names import CLASS_NAMES
from keras.models import load_model
import keras
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os  # Ensure this is imported

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Fruit Classifier")
app.mount("/images", StaticFiles(directory=os.path.join(PROJECT_ROOT, "images")), name="images")
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")
# Load models
efficientnet_model = load_model("efficientnet.h5")
mobilenet_model = load_model("mobilenet.h5")

# Initialize templates directory
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predictions": None})


@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    from PIL import Image
    # Open the image and ensure it is in RGB format
    image = Image.open(file.file).convert("RGB")
    # Resize the image to the model's input size
    image = image.resize((224, 224))
    # Normalize the image and add a batch dimension
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    mobilenet_preds = mobilenet_model.predict(image_array)[0]
    
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = keras.applications.efficientnet.preprocess_input(img_array) # Preprocess for EfficientNet
    efficientnet_preds = efficientnet_model.predict(img_array)[0]

    # Sort predictions and limit to top 10
    top_efficientnet_preds = sorted(enumerate(efficientnet_preds), key=lambda x: x[1], reverse=True)[:10]
    top_mobilenet_preds = sorted(enumerate(mobilenet_preds), key=lambda x: x[1], reverse=True)[:10]

    # Prepare predictions for rendering
    efficientnet_results = [
        {"class": CLASS_NAMES[i], "confidence": f"{prob:.2%}"} for i, prob in top_efficientnet_preds
    ]
    mobilenet_results = [
        {"class": CLASS_NAMES[i], "confidence": f"{prob:.2%}"} for i, prob in top_mobilenet_preds
    ]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predictions": {
            "efficientnet": efficientnet_results,
            "mobilenet": mobilenet_results
        }
    })

@app.get("/realtime", response_class=HTMLResponse)
def realtime_page(request: Request):
    return templates.TemplateResponse("realtime.html", {"request": request})

def generate_frames():
    camera = cv2.VideoCapture(0)
    threshold = 0.1  # Confidence threshold

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame for MobileNet
            #resized_frame = cv2.resize(frame, (224, 224))
            #image_array = np.expand_dims(resized_frame / 255.0, axis=0)
            #predictions = efficientnet_model.predict(image_array)[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            img_array = keras.preprocessing.image.img_to_array(frame)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            img_array = keras.applications.efficientnet.preprocess_input(img_array) # Preprocess for EfficientNet
            predictions = efficientnet_model.predict(img_array)[0]
            # Get top 3 predictions
            top_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:3]
            top_confidence = top_predictions[0][1]  # Confidence of the top prediction

            if top_confidence > threshold:
                # Overlay the top 3 predictions on the frame
                for idx, (class_index, confidence) in enumerate(top_predictions):
                    label = f"{CLASS_NAMES[class_index]}: {confidence:.2%}"
                    position = (10, 50 + idx * 30)  # Adjust position for each prediction
                    cv2.putText(
                        frame,
                        label,
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
            else:
                # Display 'No fruit detected'
                cv2.putText(
                    frame,
                    "No fruit detected",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
    

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return superimposed_img


@app.api_route("/grad_cam", methods=["GET", "POST"], response_class=HTMLResponse)
async def grad_cam_endpoint(request: Request, file: UploadFile = None):
    from PIL import Image

    if request.method == "GET":
        # Render the upload form
        return templates.TemplateResponse("grad_cam.html", {
            "request": request,
            "input_image": None,
            "grad_cam_image": None,
            "predictions": None
        })

    if request.method == "POST" and file:
        # Save the uploaded file temporarily
        static_dir = os.path.join(PROJECT_ROOT, "static")
        uploaded_image_path = os.path.join(static_dir, "uploaded_image.jpg")
        with open(uploaded_image_path, "wb") as f:
            f.write(await file.read())

        # Preprocess the uploaded image
        image = Image.open(uploaded_image_path).convert("RGB")
        image = image.resize((224, 224))
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = keras.applications.efficientnet.preprocess_input(img_array)  # Preprocess for EfficientNet

        # Generate predictions
        predictions = efficientnet_model.predict(img_array)[0]
        top_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:3]
        prediction_results = [
            {"class": CLASS_NAMES[i], "confidence": f"{prob:.2%}"} for i, prob in top_predictions
        ]

        # Generate Grad-CAM heatmap
        grad_cam_path = os.path.join(static_dir, "gradcam_uploaded.jpg")
        heatmap = make_gradcam_heatmap(img_array, efficientnet_model, last_conv_layer_name="top_conv")
        save_and_display_gradcam(uploaded_image_path, heatmap, cam_path=grad_cam_path)

        # URLs for the images
        input_image_url = "/static/uploaded_image.jpg"
        grad_cam_image_url = "/static/gradcam_uploaded.jpg"

        # Render the template with the image URLs and predictions
        return templates.TemplateResponse("grad_cam.html", {
            "request": request,
            "input_image": input_image_url,
            "grad_cam_image": grad_cam_image_url,
            "predictions": prediction_results
        })