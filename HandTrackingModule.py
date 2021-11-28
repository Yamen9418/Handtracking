
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelCom=1, detectoinCon=0.5, trackingcCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectoinCon
        self.trackingCon = trackingcCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert img to RGB
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks) # to check that we get value of tracking hand

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # cause we are displaying img not imgRGB
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # lm : landmarks
                # print(id,lm)
                h, w, c = img.shape  # height, width and channel
                cx, cy = int(lm.x * w), int(lm.y * h)  # convert it to int
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    if id == 4:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # draw a circle

        return lmList


def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv2.VideoCapture(0)
    #wCam, hCam = 640, 480
    #cap.set(3, wCam)
    #cap.set(4, hCam)
    detector = handDetector()  # no params since we have default params

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


if __name__ == "__main__":
    main()
