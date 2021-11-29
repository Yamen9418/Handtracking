import cv2
import time
import numpy as np

import HandTrackingModule as htm
import math
import subprocess

pTime = 0  # previous time
cTime = 0  # current time
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
vol = 0
volBar = 400
volPer = 0

detector = htm.handDetector(detectoinCon=0.7)  # to make sure its really a hand


def get_master_volume():
    proc = subprocess.Popen('/usr/bin/amixer sget Master', shell=True, stdout=subprocess.PIPE)
    amixer_stdout = str(proc.communicate()[0], 'UTF-8').split('\n')[4]
    proc.wait()

    find_start = amixer_stdout.find('[') + 1
    find_end = amixer_stdout.find('%]', find_start)

    return float(amixer_stdout[find_start:find_end])


def set_master_volume(volume):
    val = float(int(volume))
    proc = subprocess.Popen('/usr/bin/amixer sset Master ' + str(val) + '%', shell=True, stdout=subprocess.PIPE)
    proc.wait()


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # get the mid point in the line
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(lengh)

        # hand range 50 - 300
        # colume range 0-150

        vol = np.interp(length, [50, 300], [0, 150])
        volBar = np.interp(length, [50, 300], [400, 150]) # for the bar
        volPer= np.interp(length, [50, 300], [0, 100]) # vol %

        print(int(length), vol)
        set_master_volume(vol)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)  # change the color of midpoint when its pressed

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0)) # volume bar
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f' {int(volPer)} %', (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # display fps

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, ("FPS " + str(int(fps))), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # display fps

    cv2.imshow("Image", img)
    cv2.waitKey(1)
