import requests


url = "http://localhost:5000/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

print(response)
