import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import time
import keyboard

currentKey = None
key_pressed = False

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

labels = ["B", "L", "R", "S"]

def perform_action(action_key):
    global currentKey, key_pressed

    if action_key != currentKey:
        if key_pressed:
            keyboard.release(currentKey)
            key_pressed = False

        if action_key:
            keyboard.press(action_key)
            currentKey = action_key
            key_pressed = True

def release_key():
    global currentKey, key_pressed

    if key_pressed:
        keyboard.release(currentKey)
        currentKey = None
        key_pressed = False

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite)

        if index == 0:
            perform_action('s')
        elif index == 1:
            perform_action('a')
        elif index == 2:
            perform_action('d')
        elif index == 3:
            perform_action('w')
        else:
            release_key()

    else:
        release_key()

    # cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit the loop
        break

cv2.destroyAllWindows()
