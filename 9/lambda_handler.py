import os
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import json

# Set environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable MLIR optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING logs

# Load the TensorFlow Lite model interpreter & allocate memory for the tensors
# model_name = "model_2024_hairstyle_v2.tflite"
# interpreter = tflite.Interpreter(tflite_model_name)
interpreter = tflite.Interpreter(model_path="/var/task/model_2024_hairstyle_v2.tflite")
interpreter.allocate_tensors()

# Get the input and output tensor indexes
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


def download_image(url):
    """
    Downloads an image from a given URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        PIL.Image.Image: The downloaded image loaded into a PIL Image object.
    """
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    """
    Prepares an image for model input by resizing and converting it to RGB.

    Args:
        img (PIL.Image.Image): The input image to prepare.
        target_size (tuple): The target size as (width, height).

    Returns:
        PIL.Image.Image: The prepared image in RGB mode and resized to target size.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def lambda_handler(event, context):
	"""
	Lambda function handler to process an image, perform preprocessing, and run inference on a TensorFlow Lite model.

	This function downloads an image from a URL, prepares the image (resizes and normalizes it),
	performs inference using the TensorFlow Lite model, and returns the prediction as a float.

	Args:
	    event (dict): Event data passed to the Lambda function. Expected to contain a key 'url' with the image URL
	                  to be processed. If no URL is provided, a default image URL will be used.
	    context (dict): Context information passed to the Lambda function. This contains runtime details,
	                    such as the function name and execution context.

	Returns:
	    float: The model's predicted output value. This value is a float representing the model's output for the given image.
    """
	try:
		# Get the URL from the event, or use a default URL if not provided
		url = event.get('url', "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
		print(url)

		# Download and prepare the image
		img = download_image(url)
		img = prepare_image(img, target_size=(200, 200))

		# Convert the image to a NumPy array and normalize pixel values
		img = np.array(img, dtype=np.float32)
		img = np.array([img])
		img = img * 1./255

		# Perform inference & save the result
		interpreter.set_tensor(input_index, img)
		interpreter.invoke()
		pred = interpreter.get_tensor(output_index)
		result = float(pred[0][0])
		print(result)
		
		# Return the prediction
		return {
			'statusCode': 200,
			'body': json.dumps({'result': result})
		}

	except Exception as e:
		return {
			'statusCode': 500,
			'body': json.dumps({'error': str(e)})
		}
