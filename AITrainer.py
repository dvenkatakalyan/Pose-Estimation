import cv2
import mediapipe as mp
import time
import math

#self.mode-->If you put it as True it will always detect based on the model.
#self.mode-->If you put it as False--->Fast Detection + tracking
#It will always try to find new detections.
#If you put it as False it will try to detect when confidence is high it will keep tracking.
#There would be a tracking confidence and there would be a detection confidence.
#Whenever it has detected if confidence is more than 0.5(detectionCon) it will go to tracking.
#Tracking will check if tracking confidence(trackCon) is more than 0.5 it will keep tracking.
#When the tracking confidence<0.5 it will come back to detection.
#This way we do not use the heavy model again and again for detection.Instead we use detection, then tracking whenever its lost we use detection again. 
class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody#Detect only the upper part if True
        self.smooth = smooth#Feature to smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode
                                 , min_detection_confidence=0.5
                                 , min_tracking_confidence=0.5
                                 )
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
        #                              self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)#This gives us detection of pose
        #print(results.pose_landmarks)-->actual landmarks
        #landmark{x:0.514 y:1.025 z:-0.122 visibility:0.897}
        #visibility--->How visible is it?
        if self.results.pose_landmarks:#If this is present
            #results.pose_landmarks--->points
            # mpPose.POSE_CONNECTIONS-->Connections of points
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                #https://google.github.io/mediapipe/solutions/pose.html
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm) #id number 32 and landmarks eg:{x:0.514 y:1.025 z:-0.122 visibility:0.897}
                cx, cy = int(lm.x * w), int(lm.y * h) #pixel value of the landmark
                self.lmList.append([id, cx, cy])
                if draw:
                    #This will overlay on previous landmark points.
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):#Get the angle of any 3 landmarks.

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:] #[3.485,281]--->[485,281]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle at the elbow joint
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            #Drawing Lines connecting points.
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            #Drawing circles on points.
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            #displaying angle near to elbow point(x2,y2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture("1.mp4")
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)#Total 33 points
        #https://google.github.io/mediapipe/solutions/pose.html
        if len(lmList) != 0:
            print(lmList[14])#I want number 14 point(elbow) to be tracked.
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()