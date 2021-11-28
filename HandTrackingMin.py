import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 # previous time
cTime = 0 # current time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert img to RGB
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) # to check that we get value of tracking hand

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): # lm : landmarks
                #print(id,lm)
                h, w, c  = img.shape #height, width and channel
                cx, cy = int(lm.x*w), int(lm.y*h) # convert it to int
                print(id, cx, cy)
                if id==0: # start of thump finger
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED) # draw a circle





            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # cause we are displaying img not imgRGB


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255), 3) # display fps


    cv2.imshow("Image", img)
    cv2.waitKey(1)
