from sys import stdout

from inference.model.decode import create_mask, visualize
from inference.model.model import create_models, Model
from  inference.model.bg_subtraction import BackgroundSubtractorMOG2
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

crowd_model, detect_model = create_models(r'D:\CenterNet\inference\weight\vgg16_fineturning_epoch2.hdf5', r'D:\CenterNet\inference\weight\hg1stack_tfmosaic_epoch6.hdf5')
background_subtractor = BackgroundSubtractorMOG2()

model = Model(crowd_model, detect_model, background_subtractor, 0.5, 0.25, [10, 25], [0.5, 2.0])

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app, async_mode="threading")
# camera = Camera(Makeup_artist())

@socketio.on('input', namespace='/test')
def queue_inputs(input, mask, aspectRatio):
    input = input.split(",")[1]
    image_data = input
    #image_data = image_data.decode("utf-8")
    # print(mask)


    img = imread(io.BytesIO(base64.b64decode(image_data)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pts = np.array(mask, np.int32)
    env_mask = create_mask(pts)
    label, detections, diff_rate = model.predict(img, mask=env_mask)

    # img[env_mask==False] = 0
    # frame = visualize(frame, label, detections, diff_rate, model.current_classify_time, model.current_bs_time, model.current_detect_time, model.current_pred_time, display=False, output_size=(512,512))
    img = visualize(img, label, detections, diff_rate, model.current_classify_time, model.current_bs_time,
                      model.current_detect_time, model.current_pred_time, display=False, output_size=(960, 640))
    # time.sleep(0.4)
    # cv2.imwrite("next.jpg", img)

    # camera.enqueue_input(img)

    # print('QUEUED: ' + img)

    # time.sleep(0.5)

    if aspectRatio <= (960/640):
        img = cv2.resize(img, (int(640 * aspectRatio), 640))
    else:
        img = cv2.resize(img, (960, int(960 / aspectRatio)))

    _, buffer = cv2.imencode('.jpg', img)
    b = base64.b64encode(buffer)
    b = b.decode()
    image_data = "data:image/jpeg;base64," + b

    # print("OUTPUT " + image_data)
    emit('output-event', {'image_data': image_data, 'server_fps': 1000/model.current_pred_time}, namespace='/test')

    #camera.enqueue_input(base64_to_pil_image(input))

@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

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
