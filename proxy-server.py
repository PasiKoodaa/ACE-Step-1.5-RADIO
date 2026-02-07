#!/usr/bin/env python3
"""CORS Proxy for ACE-Step API"""
import requests
from flask import Flask, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ACESTEP_URL = "http://localhost:8001"

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    url = f"{ACESTEP_URL}/{path}"
    
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    resp = requests.request(
        method=request.method,
        url=url,
        headers={key: value for key, value in request.headers if key != 'Host'},
        data=request.get_data(),
        params=request.args,
        cookies=request.cookies,
        allow_redirects=False
    )
    
    response = Response(resp.content, resp.status_code)
    for key, value in resp.headers.items():
        if key.lower() not in ['content-encoding', 'transfer-encoding']:
            response.headers[key] = value
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    print("CORS Proxy running on http://localhost:8002")
    print("Proxying to ACE-Step at http://localhost:8001")
    app.run(host='0.0.0.0', port=8002)
