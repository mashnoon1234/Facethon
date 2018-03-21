import cv2
import numpy

class FaceRecognize: # This class contains all recognition algorithms encapsulated in functions
    def __init__(self, name): # Constructor / Initializer
        self.__name = name

    def recognize(self, frame, detectedFaces, recognizer, faceNames): # This function reads the command-line arguments and decides which algorithm to use
        if(self.__name == "lbph"):
            return self.__recognizeLbph(frame, detectedFaces, recognizer, faceNames)

    def __recognizeLbph(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            #face = frame[y : y + w, x : x + h]
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            faceIndex, confidence = recognizer.predict(face)
            print(confidence)
            if(faceIndex != -1):
                cv2.putText(frame, faceNames[faceIndex], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        return frame 
