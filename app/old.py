from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import numpy as np
from app.class_names import CLASS_NAMES
from keras.models import load_model
import keras
app = FastAPI(title="Fruit Classifier")

# Load models
efficientnet_model = load_model("efficientnet.h5")
mobilenet_model = load_model("mobilenet.h5")


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Fruit Classifier</title>
      </head>
      <body>
        <h1>Upload a file here</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept="image/*" required />
          <button type="submit">Upload</button>
        </form>
        <h2>Top 10 Predictions</h2>
        <div id="predictions"></div>
      </body>
    </html>
    """


@app.post("/predict")
def predict(file: UploadFile = File(...)):
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

    # Generate HTML for predictions with confidence bars
    efficientnet_html = "<h2>EfficientNet Predictions</h2><div>" + "".join(
        [
            f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
            f"<span style='width: 150px;'>{CLASS_NAMES[i]}</span>"
            f"<div style='flex: 1; max-width: 33%; background-color: #e0e0e0; height: 20px; position: relative;'>"
            f"<div style='width: {prob * 100}%; background-color: #76c7c0; height: 100%; position: absolute;'></div>"
            f"<span style='position: absolute; width: 100%; text-align: center; line-height: 20px; font-size: 12px; z-index: 1;'>{prob:.2%}</span>"
            f"</div>"
            f"</div>"
            for i, prob in top_efficientnet_preds
        ]
    ) + "</div>"

    mobilenet_html = "<h2>MobileNet Predictions</h2><div>" + "".join(
        [
            f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
            f"  <span style='width: 150px;'>{CLASS_NAMES[i]}</span>"
            f"  <div style='flex: 1; max-width: 33%; background-color: #e0e0e0; height: 20px; position: relative;'>"
            f"      <div style='width: {prob * 100}%; background-color: #76c7c0; height: 100%; position: absolute;'></div>"
            f"      <span style='position: absolute; width: 100%; text-align: center; line-height: 20px; font-size: 12px; z-index: 1;'>{prob:.2%}</span>"
            f"  </div>"
            f"</div>"
            for i, prob in top_mobilenet_preds
        ]
    ) + "</div>"

    return HTMLResponse(
        content=f"""
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Predictions</title>
      </head>
      <body>
        <h1>Predictions</h1>
        {efficientnet_html}
        {mobilenet_html}
        <a href=\"/\">Upload another file</a>
      </body>
    </html>
    """
    )

@app.get("/test", response_class=HTMLResponse)
def test():
    img_path = "./images/cauliflower.png"
    efficientnet_model = keras.models.load_model('efficientnet.h5')
    # Load and preprocess the image
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = keras.applications.efficientnet.preprocess_input(img_array) # Preprocess for EfficientNet

    # Make a prediction
    predictions = efficientnet_model.predict(img_array)[0]

    # Sort predictions and limit to top 10
    top_predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:10]

    # Generate HTML for top 10 predictions
    predictions_html = "<h2>Top 10 Predictions</h2><ul>" + "".join(
        [f"<li>{CLASS_NAMES[i]}: {prob:.2%}</li>" for i, prob in top_predictions]
    ) + "</ul>"

    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Test Prediction</title>
      </head>
      <body>
        <h1>Test Prediction</h1>
        <p>Predictions for the image at {img_path}:</p>
        {predictions_html}
        <a href=\"/\">Go back to upload page</a>
      </body>
    </html>
    """