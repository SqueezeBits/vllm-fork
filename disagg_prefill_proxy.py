import requests
from flask import Flask, request

app = Flask(__name__)

def get_hash(input_tokens):
    return hash(tuple(input_tokens))


def forward_request(url, data):
    return requests.post(url, json=data).json()    

@app.get("/v1/models")
def handle_models_request():
    return requests.get("http://localhost:8100/v1/models").json()


@app.post("/v1/completions")
def handle_completion_request():
    original_request_data = request.json
    prefill_request = dict(**original_request_data)

    hash = get_hash(original_request_data["prompt"])
    # print(f"[{hash}] received request") 


    # change max_tokens = 1 to let it only do prefill
    prefill_request["max_tokens"] = 1

    # finish prefill
    forward_request("http://localhost:8100/v1/completions", prefill_request)
    print(f"[{hash}] prefill done, proceeding to decode")


    # return decode
    result = forward_request("http://localhost:8200/v1/completions", original_request_data)
    print(f"[{hash}] decode done, request completed")


    return result
