import cv2

class FaceDetect: # This class contains all detection algorithms encapsulated in functions
    def __init__(self, name, xmlOrCfg, weights = None, gpu = None): # Constructor / Initializer
        self.__name = name
        if(self.__name == "haar"):
            self.__cascade = cv2.CascadeClassifier(xmlOrCfg)
        elif(self.__name == "lbp"):
            self.__cascade = cv2.CascadeClassifier(xmlOrCfg)
        elif(self.__name == "yolo2"):
            self.__cfg = xmlOrCfg
            self.__weights = weights
            self.__gpu = gpu 

    def detect(self, frame): # This function reads the command-line arguments and decides which algorithm to use
        if(self.__name == "haar"):
            return self.__detectHaarcascade(frame)
        elif(self.__name == "lbp"):
            return self.__detectLbpcascade(frame)
        elif(self.__name == "yolo2"):
            return self.__detectYolo2()

    def __detectHaarcascade(self, frame): # Haarcascade Face Detection
        detectedFaces = self.__cascade.detectMultiScale(frame, 1.4, 6) # Receives detected faces as an object
        for (x, y, w, h) in detectedFaces: # Iterates through the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draws rectangles around detected faces
        return frame

    def __detectLbpcascade(self, frame): # Local Binary Patterns Face Detection (better than haar according to my tests)
        detectedFaces = self.__cascade.detectMultiScale(frame, 1.1, 10) # Receives detected faces as an object
        for (x, y, w, h) in detectedFaces: # Iterates through the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draws rectangles around detected faces
        return frame


    def __detectYolo2(self): # Yolo2 Face Detection
        pass
