import cv2
from imutils.video import WebcamVideoStream # Multi-threaded Video Stream
import time # For FPS calculation

aspectRatio = 16.0 / 9.0

class Video:
    def __init__(self, url, frameWidth): # Constructor / Initializer
        self.__video = WebcamVideoStream(url).start()
        self.__frameWidth = int(frameWidth)
        self.__frameHeight = int((1.0 / aspectRatio) * self.__frameWidth)
        self.__startTime = 0.0
        self.__endTime = 0.0

    def __del__(self): # Destructor
        self.__video.stop()
        cv2.destroyAllWindows()

    def captureFrame(self): # Captures a frame from the initialized video stream
        return self.__video.read()

    def processFrame(self, frame): # Does resizing, grayscale conversion and more
        frame = cv2.UMat(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (self.__frameWidth, self.__frameHeight))
        return frame

    def startTimer(self):
        self.__startTime = time.time()

    def stopTimer(self):
        self.__endTime = time.time()

    def __showFPS(self, frame): # Private function to calculate FPS
        cv2.putText(frame, "FPS : " + str(round(1.0 / (self.__endTime - self.__startTime), 1)), (int(0.1 * self.__frameWidth), int(0.9 * self.__frameHeight)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "FPS : " + str(round(1.0 / (self.__endTime - self.__startTime), 1)), (int(0.1 * self.__frameWidth), int(0.9 * self.__frameHeight)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    def showFrame(self, frame): # Displays output frame
        self.__showFPS(frame)
        cv2.namedWindow("Facethon", cv2.WINDOW_NORMAL)
        cv2.imshow("Facethon", frame)
