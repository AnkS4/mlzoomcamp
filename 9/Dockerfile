FROM agrigorev/model-2024-hairstyle:v3

# Install TensorFlow Lite runtime
RUN curl -LO https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install pillow
RUN pip install numpy==1.23.1

# Copy the lambda handler script to the container
COPY lambda_handler.py .

# Set the default command to execute the handler script
CMD ["lambda_handler.lambda_handler"]
