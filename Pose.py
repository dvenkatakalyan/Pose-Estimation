import cv2
import numpy as np
import time
import PoseModule as pm
from AITrainer import poseDetector
cap = cv2.VideoCapture("1.mp4")

detector = pm.poseDetector()
count = 0 #counting number of curls
dir = 0 
#dir=0--->when dumbell is going up
#dir=1--->when dumbell is going down
#We will consider a full curl only if it does both of these.
pTime = 0 #Previous time
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'),30,(1280, 720))
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("AiTrainer/test.jpg")
    img = detector.findPose(img, False)#False-->only 3 points rest will be gone.
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        # Right Arm--->12,14,16
        #https://google.github.io/mediapipe/solutions/pose.html
        angle = detector.findAngle(img, 12, 14, 16)
        # # Left Arm-->11,13,15
        #angle = detector.findAngle(img, 11, 13, 15,False)
        #We are getting angles from 210 to 310 when the person is lifting the dumbell up and down.
        #we are converting range from (210-310) to (0-100)
        per = np.interp(angle, (210, 310), (0, 100))
        #we are converting range from (210-310) to (650-100) for the bar
        bar = np.interp(angle, (220, 310), (650, 100))
        # print(angle, per)

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:#dumbell going up
                count += 0.5 #we will add 0.5 to count for up
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:#dumbell going down
                count += 0.5 #we will add 0.5 to count for down
                dir = 0
        print(count)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 0, 0), 25)

    cTime = time.time() #Current time
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)
    out.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)