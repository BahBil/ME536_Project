import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle5 as pickle
mp_drawing = mp.python.solutions.drawing_utils
mp_hands = mp.python.solutions.hands

script_dir = os.path.dirname(__file__)
print(script_dir)

HandData = np.empty([21,2])

TotalDataCounter = 0
TotalData = np.empty([150,21,2])

for handshape in ("sci_","roc_","pap_","spo_","rep_"):
    for adresscounter in range(30):
        rel_path = "Gestures_Named6\\" + handshape + str(adresscounter) + ".png"
        abs_file_path = os.path.join(script_dir, rel_path)
       # print(abs_file_path)

        # For static images:
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5)
        for idx, file in enumerate((abs_file_path,)):
          # Read an image, flip it around y-axis for correct handedness output (see
          # above).
          #print(idx)
          #print(file)
          image = cv2.flip(cv2.imread(file), cv2.IMREAD_COLOR)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          #plt.figure()
          #plt.title('original image')
          #plt.imshow(image)
          #plt.show()
          # Convert the BGR image to RGB before processing.
          results = hands.process(image)

          # Print handedness and draw hand landmarks on the image.
          #print('Handedness:', results.multi_handedness)
          if not results.multi_hand_landmarks:
            continue
          image_hight, image_width, _ = image.shape
          annotated_image = image.copy()
          for hand_landmarks in results.multi_hand_landmarks:
            #print('hand_landmarks:', hand_landmarks)
            #print(
            #    f'Index finger tip coordinates: (',
            #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            #)
            #HandData = []
            for indexx , handtips in  enumerate(mp_hands.HandLandmark):
                HandData[indexx,0] = hand_landmarks.landmark[handtips].x
                HandData[indexx,1] = hand_landmarks.landmark[handtips].y

            #print(hand_landmarks.landmark[:].x)
            #print(hand_landmarks.landmark[:].y)
            TotalData[TotalDataCounter]=HandData
            TotalDataCounter += 1

            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          cv2.imwrite(
              '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        hands.close()
       
#print(TotalData[0:5])
plt.figure()
plt.title('original image')
plt.imshow(annotated_image)
plt.show()

with open('objs6.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(TotalData, f)


"""
# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
"""