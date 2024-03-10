import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

DIR = r'out_images'
CATEGORIES = os.listdir(DIR)

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Wrong path:', path)
    else:
        new_arr = cv2.resize(img, (256, 256))
        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(-1, 256, 256, 1)
        return new_arr

model = keras.models.load_model('NoJacket.model')
prediction = model.predict([image('test/test_jacket.jpg')])
print(CATEGORIES[prediction.argmax()])