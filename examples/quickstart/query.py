import requests

result = requests.post(
   "http://127.0.0.1:3000/classify",
   headers={"content-type": "application/json"},
   data="[[5.9, 3, 5.1, 1.8]]",
).text

print(result)