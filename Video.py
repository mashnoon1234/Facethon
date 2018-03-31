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
    #self.__lightCorrection = cv2.createCLAHE(60, (3, 3))

    def __del__(self): # Destructor
        self.__video.stop()
        cv2.destroyAllWindows()

    def captureFrame(self): # Captures a frame from the initialized video stream
        return self.__video.read()
    
    def __transferFrameGPU(self, frame):
        return cv2.UMat(frame)
    
    def __resizeFrame(self, frame):
        return cv2.resize(frame, (self.__frameWidth, self.__frameHeight))
    
    def __convertFrameToGrayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def __correctFrameContrastInGrayscale(self, frame):
        cv2.equalizeHist(frame, frame)
        return frame
    
    def processFrame(self, frame): # Does resizing, grayscale conversion and more
        frame = self.__transferFrameGPU(frame)
        frame = self.__resizeFrame(frame)
        #b, g, r = cv2.split(frame)
        #self.__lightCorrection.apply(b, b)
        #self.__lightCorrection.apply(g, g)
        #self.__lightCorrection.apply(r, r)
        #frame = cv2.merge((b, g, r))
        #frame = self.__transferFrameGPU(frame)
        #frame = self.__convertFrameToGrayscale(frame)
        #self.__lightCorrection.apply(frame, frame)
        #frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def startTimer(self):
        self.__startTime = time.time()

    def stopTimer(self):
        self.__endTime = time.time()
    
    def __writeTextToFrame(self, frame, text, position):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType = cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType = cv2.LINE_AA)
        return frame
    
    def __showFPS(self, frame): # Private function to calculate FPS
        text = "FPS : " + str(round(1.0 / (self.__endTime - self.__startTime), 1))
        position = (int(0.1 * self.__frameWidth), int(0.9 * self.__frameHeight))
        return self.__writeTextToFrame(frame, text, position)

    def showFrame(self, frame): # Displays output frame
        frame = self.__showFPS(frame)
        cv2.namedWindow("Facethon", cv2.WINDOW_NORMAL)
        cv2.imshow("Facethon", frame)
