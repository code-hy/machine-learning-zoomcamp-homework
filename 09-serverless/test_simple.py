import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
data = {"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']:.4f}")
