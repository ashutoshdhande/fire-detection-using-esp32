from lib2to3.pgen2 import token
import urllib.request
import cv2 as cv
import numpy as np
from PIL import Image
from keras import models
from keras.preprocessing import image
import time
import os
import telegram_send

# Enviroment vars
# os.environ['token']='5230365611:AAHr8LeyMYioTE3amGeTxlXp6BJCGnmuXY0'
api_token = '5230365611:AAHr8LeyMYioTE3amGeTxlXp6BJCGnmuXY0'

#load the model
model = models.load_model('fire_model.h5')
url='http://192.168.27.84/capture?'
frame = None
key = None

print("Streaming started....")
while True:
    img_res = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(img_res.read()), dtype=np.uint8)
    # print("imgNp ", imgNp)
    frame = cv.imdecode(imgNp, -1)
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((224, 224))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)
    if prediction == 0:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        print("Fire detected")
        telegram_send.send(messages=["ALERT!!!", "Fire detected!!!"])
        # print(probabilities[prediction])
        time.sleep(2)
    cv.imshow('Window', frame)
    key = cv.waitKey(500)
    if key == (ord('q')):
        break
cv.destroyAllWindows()
print("Stream Ended")