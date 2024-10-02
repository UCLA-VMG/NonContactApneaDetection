import numpy as np
import os
import tqdm
from tqdm import tqdm

import matplotlib.pyplot as plt
import imageio
import scipy
import cv2
import decord
from mtcnn import MTCNN
import tensorflow as tf # TODO: uncomment
import PIL

class ThermalFaceDetector:
    def __init__(self, model_path, verbose=False):
        self.verbose = verbose
        # Load the TFLite model and allocate tensors.
        self.nn_detector = tf.lite.Interpreter(model_path=model_path)
        self.nn_detector.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.nn_detector.get_input_details()
        self.output_details = self.nn_detector.get_output_details()
        if verbose: print(self.input_details, self.output_details)

    def detect_face(self, img, buffer=(0.0, 0.0)):
        # get width (x) and height (y) buffer
        x_buffer, y_buffer = buffer
        # get crop coords using detector
        img = cv2.resize(img, (192,192))[np.newaxis]
        print(img.shape)
        # img = np.expand_dims(img, 0)
        self.nn_detector.set_tensor(self.input_details[0]['index'], img)
        self.nn_detector.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        output_data = self.nn_detector.get_tensor(self.output_details[0]['index'])
        y1, x1, y2, x2 = output_data.squeeze()[0]
        if self.verbose: print(f"BBox : {y1},{x1},{y2},{x2}")
        x = int(x1*img.shape[1]) 
        y = int(y1*img.shape[0])
        w = int((x2-x1)*img.shape[1])
        h = int((y2-y1)*img.shape[0])
        # update crop coords to apply buffer
        x = max(int(x - (w * x_buffer )), 0)
        y = max(int(y - (h * y_buffer)), 0)
        w = int(w * (1 + 2 * x_buffer ))
        h = int(h * (1 + 2 * y_buffer))
        # return updated crop coords
        print(x, y, w, h)
        return x, y, w, h

    def crop_data(self, data, buffer=(0.0, 0.0)):
        # We assume that the face does not move during the video, so we take coords from the first frame
        frame = data[0]
        # if frame dtype is uint16, convert to uint8 for crop coord detection
        if data.dtype == np.uint16:
            frame = (frame.astype('float') / (2**16 - 1) * 255).astype('uint8')
        # if frame is single channel, convert to triple channel by stacking
        if (frame.shape[2] == 1):
            frame = np.concatenate([frame, frame, frame], axis=2)
        # use detector to get crop coords
        x, y, w, h = self.detect_face(frame, buffer)
        # crop data
        data = data[:, y:y+h, x:x+w, :] 
        # return data
        print(data.shape)
        return data

class FaceDetector:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.nn_detector = MTCNN()

    def detect_face(self, img, buffer=(0.0, 0.0), face_index=0):
        # get width (x) and height (y) buffer
        x_buffer, y_buffer = buffer
        # get crop coords using detector
        x, y, w, h = self.nn_detector.detect_faces(img)[face_index]['box']
        # update crop coords to apply buffer
        x = max(int(x - (w * x_buffer )), 0)
        y = max(int(y - (h * y_buffer)), 0)
        w = int(w * (1 + 2 * x_buffer ))
        h = int(h * (1 + 2 * y_buffer))
        # return updated crop coords
        return x, y, w, h

    def crop_data(self, data, buffer=(0.0, 0.0)):
        # We assume that the face does not move during the video, so we take coords from the first frame
        frame = data[0]
        # if frame is single channel, convert to triple channel by stacking
        if (frame.shape[2] == 1):
            frame = np.concatenate([frame, frame, frame], axis=2)
        # use detector to get crop coords
        x, y, w, h = self.detect_face(frame, buffer, face_index=0)
        # crop data
        data = data[:, y:y+h, x:x+w, :] 
        # return data
        return data