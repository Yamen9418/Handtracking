import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


pTime = 0  # previous time
cTime = 0  # current time
cap = cv2.VideoCapture(0)
detector = htm.handDetector()  # no params since we have default params

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # display fps

    cv2.imshow("Image", img)
    cv2.waitKey(1)