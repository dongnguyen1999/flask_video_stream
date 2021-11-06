from sys import stdout

from inference.utils import create_mask, visualize, get_masked_img
from inference.models.model import Model
from inference.models.frame_difference import FrameDiffEstimator
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
import numpy as np
import base64
import io
from imageio import imread
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import json

model = Model('crowd_model', 'detect_model')

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app, async_mode="threading")


# camera = Camera(Makeup_artist())

@socketio.on('input', namespace='/test')
def queue_inputs(input, mask, config):
    global model

    input = input.split(",")[1]

    config_dict = json.loads(config)

    reset_session = config_dict['reset_session']
    aspect_ratio = config_dict['aspect_ratio']

    image_data = input
    # image_data = image_data.decode("utf-8")
    # print(mask)

    img = imread(io.BytesIO(base64.b64decode(image_data)))

    pts = np.array(mask, np.int32)
    env_mask = create_mask(pts)

    if reset_session:
        model.reset_session()

    result = model.predict(img, mask=env_mask, config=config_dict)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = get_masked_img(img, env_mask)


    img, result = visualize(img, result, model.current_classify_time, model.current_bs_time, model.current_count_time, model.current_pred_time, config=config_dict)

    # plt.imshow(img)
    # plt.show()

    _, buffer = cv2.imencode('.jpg', img)
    b = base64.b64encode(buffer)
    b = b.decode()
    image_data = "data:image/jpeg;base64," + b

    # print(json.dumps(result))

    # print("OUTPUT " + image_data)
    emit('output-event', {'image_data': image_data, 'result': json.dumps(result)}, namespace='/test')

    # camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")

@app.route('/change_model', methods=['POST'])
def change_model():
    try:
        global model
        data = request.form.to_dict(flat=False)
        model.make_model(data['crowd_model'][0], data['count_model'][0])
        response = {
            'status': 200,
            'message': 'Change model success',
            'error': False,
        }
    except:
        response = {
            'status': 500,
            'message': 'Internal server error',
            'error': True,
        }
    return json.dumps(response)

@app.route('/')
def index():
    global model
    """Video streaming home page."""
    model_data = {
       'crowd_model': model.crowd_model_name,
       'count_model': model.count_model_name,
       'config': {
           'count_conf_threshold': model.count_conf_threshold,
           'classify_conf_threshold': model.classify_conf_threshold,
           'count_thresholds': model.count_thresholds,
           'crowd_thresholds': model.crowd_thresholds,
           'heatmap_only': model.heatmap_only,
       }
    }
    return render_template('index.html', model_data=model_data)


# def gen():
#     """Video streaming generator function."""
#
#     app.logger.info("starting to generate frames!")
#     while True:
#         frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())
#
#         print(type(frame))
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':



    socketio.run(app)
