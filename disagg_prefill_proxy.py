import requests
from flask import Flask, request

app = Flask(__name__)

def forward_request(url, data):
    return requests.post(url, json=data).json()    

@app.post('/v1/completions')
def handle_request():
    original_request_data = request.json
    prefill_request = dict(**original_request_data)

    # change max_tokens = 1 to let it only do prefill
    prefill_request['max_tokens'] = 1

    # finish prefill
    for _ in forward_request('http://localhost:8100/v1/completions', prefill_request):
        continue

    print("Prefill done. proceeding to decode.")

    # return decode
    result = forward_request('http://localhost:8200/v1/completions', original_request_data)

    return result
