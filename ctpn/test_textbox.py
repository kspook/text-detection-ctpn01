import cv2
import numpy as np
import os
import base64
import json
import requests
import tensorflow as tf

image = r"/home/kspook/text-detection-ctpn/data/demo/006.jpg"
URL="http://localhost:9001/v1/models/ctpn:predict" 
#URL = "http://{HOST:port}/v1/models/<modelname>/versions/1:classify" 
headers = {"content-type": "application/json"}
image_content = base64.b64encode(open(image,'rb').read()).decode("utf-8")
body={
     "signature_name": "predict_images_post",
     "inputs": [
                image_content
      ]        
}
r= requests.post(URL, data=json.dumps(body), headers = headers) 
print(r.text)
