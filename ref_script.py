import cv2
import numpy as np
from PIL import Image
from keras import models
from keras.preprocessing import image
import time

# s = serial.Serial('COM1', 9600)

#load the model
model = models.load_model('fire_model.h5')
video = cv2.VideoCapture(0)

while True:
  _, frame = video.read()
  im = Image.fromarray(frame, 'RGB')
  im = im.resize((224, 224))
  img_array = image.img_to_array(im)
  img_array = np.expand_dims(img_array, axis=0) / 255
  probabilities = model.predict(img_array)[0]
  prediction = np.argmax(probabilities)
  if prediction == 0:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    print("Fire detected")
    # s.write(b'1')
    print(probabilities[prediction])
    time.sleep(2)
  cv2.imshow('capturing', frame)
#   s.write(b'0')
  print("No fire")
  key = cv2.waitKey(1)
  if key == ord('q'):
    break
video.release()
cv2.destroyAllWindows()