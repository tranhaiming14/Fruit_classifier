from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow.keras.models import load_model
from app.class_names import CLASS_NAMES

app = FastAPI(title="Fruit Classifier")

# Load models
#efficientnet_model = load_model("efficientnet.h5")
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

    # Get predictions from the model
    mobilenet_preds = mobilenet_model.predict(image_array)[0]

    # Get top 10 predictions
    top_indices = np.argsort(mobilenet_preds)[-10:][::-1]
    top_predictions = [(CLASS_NAMES[i], mobilenet_preds[i]) for i in top_indices]

    # Generate HTML for predictions
    predictions_html = "<ul>" + "".join(
        [f"<li>{name}: {prob:.2%}</li>" for name, prob in top_predictions]
    ) + "</ul>"

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
        <h1>Top 10 Predictions</h1>
        {predictions_html}
        <a href=\"/\">Upload another file</a>
      </body>
    </html>
    """
    )