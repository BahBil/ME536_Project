import numpy as np
#import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
#from tensorflow.keras import layers
import scipy as sp
#from scipy.optimize import minimize
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

sci_counter = 0
roc_counter = 0
spo_counter = 0
pap_counter = 0
rep_counter = 0
ran_counter = 0




while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    key = cv2.waitKey(1)
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif key%256 == 97:
        img_name = "sci_{}.png".format(sci_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        sci_counter += 1
    elif key%256 == 98:
        img_name = "roc_{}.png".format(roc_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        roc_counter += 1
    elif key%256 == 99:
        img_name = "spo_{}.png".format(spo_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        spo_counter += 1
    elif key%256 == 100:
        img_name = "pap_{}.png".format(pap_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        pap_counter += 1
    elif key%256 == 101:
        img_name = "rep_{}.png".format(rep_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        rep_counter += 1
    elif key%256 == 102:
        img_name = "ran_{}.png".format(ran_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        ran_counter += 1

cam.release()

cv2.destroyAllWindows()