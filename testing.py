import requests

url = "http://209.38.120.5:8000/predict/"
file_path = "images/euca.jpg"

with open(file_path, "rb") as img:
    files = {"file": img}
    response = requests.post(url, files=files)

print(response.json()) 