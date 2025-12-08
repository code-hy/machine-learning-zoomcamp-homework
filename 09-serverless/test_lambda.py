import requests
import json

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

data = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

print("Sending request to Lambda function...")
response = requests.post(url, json=data)

print("\n" + "="*50)
print("LAMBDA FUNCTION RESPONSE")
print("="*50)
print(f"Status Code: {response.status_code}")
print(f"\nRaw Response Text:")
print(response.text)
print(f"\nParsed JSON:")
result = response.json()
print(json.dumps(result, indent=2))
print(f"\nPrediction Value: {result['prediction']}")
print("="*50)
