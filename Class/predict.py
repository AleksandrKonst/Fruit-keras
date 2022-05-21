import os

import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DIR = r'Training'

dirfiles = os.listdir(DIR)

fullpaths = map(lambda name: os.path.join(DIR, name), dirfiles)

CATEGORIES = []

for file in fullpaths:
  CATEGORIES.append(os.path.basename(file))

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Wrong path:', path)
    else:
        new_arr = cv2.resize(img, (60, 60))
        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(-1, 60, 60, 1)
        return new_arr

model = keras.models.load_model('Model.model')
prediction = model.predict([image('first.jpg')])
print(CATEGORIES[prediction.argmax()])
img = Image.open(r'first.jpg')
images = [img]
prediction = model.predict([image('second.jpg')])
print(CATEGORIES[prediction.argmax()])
img = Image.open(r'second.jpg')
images.append(img)

plt.figure(figsize=(4,4))
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap=plt.cm.binary)
plt.show()