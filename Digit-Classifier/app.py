from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import logging
import anvil.server
import io
from PIL import Image, ImageDraw
import base64
import PIL
import os

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the saved models
cnn_model_path = 'content_cnn/CNN'
vit_model_path = 'content_vit/ViT'
try:
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    app.logger.info("CNN Model loaded successfully.")

    vit_model = tf.keras.models.load_model(vit_model_path)
    app.logger.info("ViT Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading the model: {e}")

# Function to preprocess input for ViT
def preprocess_input_for_vit(x_data, n=7, m=7, block_size=16):
    ndata = x_data.shape[0]  # Number of samples
    x_ravel = np.zeros((ndata, n*m, block_size))
    for img in range(ndata):
        ind = 0
        for row in range(n):
            for col in range(m):
                x_ravel[img, ind, :] = x_data[img, (row*4):((row+1)*4), (col*4):((col+1)*4)].ravel()
                ind += 1
    pos_feed = np.array([list(range(n*m))] * ndata)
    return x_ravel, pos_feed


def draw_grid(img, grid_size=(7, 7), line_width=8):
    """
    Draws grid lines on the image.
    :param img: PIL Image object
    :param grid_size: Tuple indicating the number of blocks in each dimension (n, m)
    :param line_width: Width of the grid lines
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    n, m = grid_size
    
    # Calculate block size
    block_w, block_h = w / m, h / n
    
    # Draw vertical lines
    for i in range(1, m):
        draw.line([(i * block_w, 0), (i * block_w, h)], fill=217, width=line_width)
    
    # Draw horizontal lines
    for i in range(1, n):
        draw.line([(0, i * block_h), (w, i * block_h)], fill=217, width=line_width)


def apply_rounded_corners(img, radius=20):
    """
    Applies rounded corners to an image.
    :param img: PIL Image object
    :param radius: Radius for the rounded corners
    """
    circle = Image.new('L', (radius * 2, radius * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)
    alpha = Image.new('L', img.size, 255)
    
    w,h = img.size
    alpha.paste(circle.crop((0, 0, radius, radius)), (0, 0))
    alpha.paste(circle.crop((0, radius, radius, radius * 2)), (0, h - radius))
    alpha.paste(circle.crop((radius, 0, radius * 2, radius)), (w - radius, 0))
    alpha.paste(circle.crop((radius, radius, radius * 2, radius * 2)), (w - radius, h - radius))
    img.putalpha(alpha)
    return img


@anvil.server.callable
def predict(file):
    try:
        # Check if the provided object has the get_name() method (Anvil file object)
        if hasattr(file, 'get_name'):
            # Access the file name from the Anvil file object
            file_name = file.get_name()
        else:
            # It's not an Anvil file object, assume it's a regular file object
            file_name = file.name
        
        # Use os.path.splitext to get the file extension
        _, file_extension = os.path.splitext(file_name)

        # Check if the file has the correct extension
        if file_extension.lower() != '.csv':
            return {'error': 'Invalid file format. Expected a CSV file.'}

        # Convert the Anvil file to a file-like object
        with io.BytesIO(file.get_bytes()) as file_obj:
            # First, attempt to read without assuming a header
            df = pd.read_csv(file_obj, header=None)
            if df.shape != (28, 28):
                # If the shape is not as expected, it might have a header
                file_obj.seek(0)
                df = pd.read_csv(file_obj, header=0)
                
                # Check again if the shape is correct after assuming a header
                if df.shape != (28, 28):
                    return {'error': 'Invalid CSV format. Expected 28 rows and 28 columns, with or without a header.'}
        
        input_data = df.values
        if input_data.max() > 1:
            input_data /= 255.0  # Normalize if necessary
        
        # Reshape for CNN model and predict
        input_data_cnn = input_data.reshape(-1, 28, 28, 1)  # For CNN
        cnn_predictions = cnn_model.predict(input_data_cnn)
        cnn_predicted_class = np.argmax(cnn_predictions, axis=1)
        cnn_confidence = np.max(cnn_predictions, axis=1)

        # Preprocess and reshape input data for ViT model
        input_data_vit, pos_feed_vit = preprocess_input_for_vit(input_data.reshape(-1, 28, 28), n=7, m=7, block_size=16)
        vit_predictions = vit_model.predict([input_data_vit, pos_feed_vit])
        vit_predicted_class = np.argmax(vit_predictions, axis=1)
        vit_confidence = np.max(vit_predictions, axis=1)
        
        # Convert the first row of CSV data to an image for visualization
        img_array = input_data_cnn[0].reshape(28, 28) * 255

        # Image for CNN
        img_cnn = Image.fromarray(img_array).convert("L")
        img_cnn = img_cnn.resize((512, 512), Image.Resampling.LANCZOS)  # Resize for better visibility
        img_cnn = apply_rounded_corners(img_cnn)  # Apply rounded corners

        # Convert to data URL for CNN
        buffered_cnn = io.BytesIO()
        img_cnn.save(buffered_cnn, format="PNG")
        img_str_cnn = base64.b64encode(buffered_cnn.getvalue()).decode()
        data_url_cnn = f"data:image/png;base64,{img_str_cnn}"

        # Image for ViT
        img_vit = Image.fromarray(img_array).convert("L")
        img_vit = img_vit.resize((512, 512), Image.Resampling.LANCZOS)  # Resize for better visibility
        draw_grid(img_vit, grid_size=(7, 7))  # Draw grid lines for ViT blocks
        img_vit = apply_rounded_corners(img_vit)  # Apply rounded corners

        # Convert to data URL for ViT
        buffered_vit = io.BytesIO()
        img_vit.save(buffered_vit, format="PNG")
        img_str_vit = base64.b64encode(buffered_vit.getvalue()).decode()
        data_url_vit = f"data:image/png;base64,{img_str_vit}"

        response_data = []

        for cnn_pred, cnn_conf, vit_pred, vit_conf in zip(cnn_predicted_class, cnn_confidence, vit_predicted_class, vit_confidence):
            response_data.append({
                "model": "CNN",
                "predicted_class": int(cnn_pred),
                "confidence": float(cnn_conf),
                "image_data_url": data_url_cnn 
            })
            response_data.append({
                "model": "ViT",
                "predicted_class": int(vit_pred),
                "confidence": float(vit_conf),
                "image_data_url": data_url_vit
            })

        return response_data
    except Exception as e:
        return {'error': f'Failed to process the file: {str(e)}'}

@app.route('/predict', methods=['POST'])
def http_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            df = pd.read_csv(file, header=None)
            if df.shape == (28, 28):
                input_data = df.values
            elif df.shape == (29, 28):  # Assuming one row for header
                app.logger.info("Detected header in CSV file.")
                file.seek(0)  # Reset file pointer to the beginning
                df = pd.read_csv(file, header=0)
                if df.shape != (28, 28):
                    return jsonify({'error': 'Invalid CSV format after header removal'}), 400
                input_data = df.values
            else:
                return jsonify({'error': 'Invalid CSV format. Expected 28 rows and 28 columns, with or without header'}), 400
            
            if input_data.max() > 1:
                input_data /= 255.0  # Normalize if necessary
            
            # Reshape for CNN model and predict
            input_data_cnn = input_data.reshape(-1, 28, 28, 1)  # For CNN
            cnn_predictions = cnn_model.predict(input_data_cnn)
            cnn_predicted_class = np.argmax(cnn_predictions, axis=1)
            cnn_confidence = np.max(cnn_predictions, axis=1)

            # Preprocess and reshape input data for ViT model
            input_data_vit, pos_feed_vit = preprocess_input_for_vit(input_data.reshape(-1, 28, 28), n=7, m=7, block_size=16)
            vit_predictions = vit_model.predict([input_data_vit, pos_feed_vit])
            vit_predicted_class = np.argmax(vit_predictions, axis=1)
            vit_confidence = np.max(vit_predictions, axis=1)
            
            # Convert the first row of CSV data to an image for visualization
            img_array = input_data_cnn[0].reshape(28, 28) * 255

            # Image for CNN
            img_cnn = Image.fromarray(img_array).convert("L")
            img_cnn = img_cnn.resize((512, 512), Image.Resampling.LANCZOS)  # Resize for better visibility
            img_cnn = apply_rounded_corners(img_cnn)  # Apply rounded corners

            # Convert to data URL for CNN
            buffered_cnn = io.BytesIO()
            img_cnn.save(buffered_cnn, format="PNG")
            img_str_cnn = base64.b64encode(buffered_cnn.getvalue()).decode()
            data_url_cnn = f"data:image/png;base64,{img_str_cnn}"

            # Image for ViT
            img_vit = Image.fromarray(img_array).convert("L")
            img_vit = img_vit.resize((512, 512), Image.Resampling.LANCZOS)  # Resize for better visibility
            draw_grid(img_vit, grid_size=(7, 7))  # Draw grid lines for ViT blocks
            img_vit = apply_rounded_corners(img_vit)  # Apply rounded corners

            # Convert to data URL for ViT
            buffered_vit = io.BytesIO()
            img_vit.save(buffered_vit, format="PNG")
            img_str_vit = base64.b64encode(buffered_vit.getvalue()).decode()
            data_url_vit = f"data:image/png;base64,{img_str_vit}"

            response_data = []

            for cnn_pred, cnn_conf, vit_pred, vit_conf in zip(cnn_predicted_class, cnn_confidence, vit_predicted_class, vit_confidence):
                response_data.append({
                    "model": "CNN",
                    "predicted_class": int(cnn_pred),
                    "confidence": float(cnn_conf),
                    "image_data_url": data_url_cnn 
                })
                response_data.append({
                    "model": "ViT",
                    "predicted_class": int(vit_pred),
                    "confidence": float(vit_conf),
                    "image_data_url": data_url_vit
                })

            return jsonify(response_data)
        except Exception as e:
            app.logger.error(f"Error processing the file: {e}")
            return jsonify({'error': str(e)}), 400

anvil_server_uplink_key = "uplink_key"
anvil.server.connect(anvil_server_uplink_key)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
