from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
import pandas as pd
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss

app = Flask(__name__, template_folder='templates')

# Load configuration for image paths
with open('image_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {int(k): v for k, v in json_dict.items()}
LenDictPath = len(DictImagePath)
bin_file = 'faiss_normal_ViT.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")

@app.route('/home')
@app.route('/')
def thumbnailimg():
    index = request.args.get('index', default=0, type=int)
    img_per_index = 100
    first_index = index * img_per_index
    last_index = min(first_index + img_per_index, LenDictPath)

    pagefile = [{'imgpath': DictImagePath[idx], 'id': idx} for idx in range(first_index, last_index)]
    data = {'num_page': (LenDictPath + img_per_index - 1) // img_per_index, 'pagefile': pagefile}
    
    return render_template('home.html', data=data)

@app.route('/imgsearch')
def image_search():
    id_query = request.args.get('imgid', default=0, type=int)
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=50)
    pagefile = [{'imgpath': path, 'id': int(id)} for path, id in zip(list_image_paths, list_ids)]

    data = {'num_page': (LenDictPath + 99) // 100, 'pagefile': pagefile}
    return render_template('home.html', data=data)

@app.route('/textsearch')
def text_search():
    text_query = request.args.get('textquery', default="", type=str)
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=50)
    pagefile = [{'imgpath': path, 'id': int(id)} for path, id in zip(list_image_paths, list_ids)]

    data = {'num_page': (LenDictPath + 99) // 100, 'pagefile': pagefile}
    return render_template('home.html', data=data)

@app.route('/get_img')
def get_img():
    fpath = request.args.get('fpath', default='./static/images/404.jpg', type=str)
    if not os.path.exists(fpath):
        fpath = "./static/images/404.jpg"
    
    img = cv2.imread(fpath)
    img = cv2.resize(img, (1280, 720))
    image_name = os.path.join(*fpath.split('/')[-2:])
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
    
    ret, jpeg = cv2.imencode('.jpg', img)
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
