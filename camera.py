import threading
import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
import time


class Camera(object):
    def __init__(self):
        self.queue = []


    def enqueue_input(self, input):
        self.queue.append(input)

    def dequeue_input(self):
        if len(self.queue) == 0:
            return None
        return self.queue.pop(0)

    def reset_queue(self):
        self.queue = []


