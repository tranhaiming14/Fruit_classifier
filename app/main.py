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

app = FastAPI(title="Fruit Classifier")

# Load models
efficientnet_model = load_model("efficientnet.h5")
mobilenet_model = load_model("mobilenet.h5")

# Initialize templates directory
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    #Seperate preprocessing for EfficientNet
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
    threshold = 0.5  # Confidence threshold

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess the frame for mobilenet
            #resized_frame = cv2.resize(frame, (224, 224))
            #image_array = np.expand_dims(resized_frame / 255.0, axis=0)
            #predictions = efficientnet_model.predict(image_array)[0]
            
            # Preprocess the frame for EfficientNet
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