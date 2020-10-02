import tensorflow as tf
import numpy as np
import cv2


class Model:
    def __init__(self, model_path, img_path):
        self.model_path = model_path
        self.img_path = img_path
        self.model = tf.keras.models.load_model(model_path)
    def make_predict(self):
        img_array1 = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        img_array2 = cv2.resize(img_array1, (28, 28))
        img_data = cv2.bitwise_not(img_array2).reshape(1, 28, 28)
        prediction = self.model.predict(img_data)
        return prediction


