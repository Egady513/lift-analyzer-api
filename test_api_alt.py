import http.client
import json

print("Testing API with http.client...")
conn = http.client.HTTPConnection("127.0.0.1", 5000)
conn.request("GET", "/test")
response = conn.getresponse()
print(f"Status: {response.status} {response.reason}")
data = json.loads(response.read().decode())
print(f"Response data: {data}") 