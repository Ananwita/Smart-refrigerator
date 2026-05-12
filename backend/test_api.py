import requests

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Change this to your image path
image_path = "test.jpg"

# Send request
with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())