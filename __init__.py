import requests

response = requests.post(
    "http://localhost:8000/coding/invoke",
    json={"question": "What is this simbolo about?"}
)
print(response.json())

