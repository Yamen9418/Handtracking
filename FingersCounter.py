import cv2
import time
import os  # to store finger images
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "fingers"
myList = sorted(os.listdir(folderPath))  # to list all images
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectoinCon=0.75)

tipIds = [4, 8, 12, 16, 20]  # finger tips based on handmarks image

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # open finger based on handmarks
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):  # just for 4 fingers "not the thumb"
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # open finger based on handmarks
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)  # to find how many ones we have in our list
        # print(totalFingers)
        # print(lmList)

        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, ("FPS " + str(int(fps))), (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # display fps

    cv2.imshow("Image", img)
    cv2.waitKey(1)
