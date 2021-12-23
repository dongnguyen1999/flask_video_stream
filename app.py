from sys import stdout
from inference.utils import create_mask, visualize, get_masked_img
from inference.models.model import Model
import logging
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from imageio import imread
import json

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app, async_mode="threading")


@socketio.on('input', namespace='/test')
def queue_inputs(input, mask, config):
    global model

    input = input.split(",")[1]

    config_dict = json.loads(config)

    reset_session = config_dict['reset_session']

    image_data = input

    img = imread(io.BytesIO(base64.b64decode(image_data)))

    pts = np.array(mask, np.int32)
    env_mask = create_mask(pts)

    if reset_session:
        model.reset_session()

    pred_output = model.predict(img, mask=env_mask, config=config_dict)

    if pred_output is not None:
        out_img, result = pred_output
        out_img = get_masked_img(out_img, env_mask)
        out_img, result = visualize(out_img, result, model.current_classify_time, model.current_bs_time, model.current_count_time, model.current_pred_time, config=config_dict)

        # plt.imshow(img)
        # plt.show()

        _, buffer = cv2.imencode('.jpg', out_img)
        b = base64.b64encode(buffer)
        b = b.decode()
        image_data = "data:image/jpeg;base64," + b

        # print(json.dumps(result))

        # print("OUTPUT " + image_data)
        emit('output-event', {'image_data': image_data, 'result': json.dumps(result)}, namespace='/test')

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

if __name__ == '__main__':
    model = Model('crowd_model', 'detect_model')
    socketio.run(app, use_reloader=False)
