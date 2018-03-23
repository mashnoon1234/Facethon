import cv2
import numpy

class FaceRecognize: # This class contains all recognition algorithms encapsulated in functions
    def __init__(self, name): # Constructor / Initializer
        self.__name = name

    def recognize(self, frame, detectedFaces, recognizer, faceNames): # This function reads the command-line arguments and decides which algorithm to use
        if(self.__name == "lbph"):
            return self.__recognizeLbph(frame, detectedFaces, recognizer, faceNames)
        elif(self.__name == "fisher"):
            return self.__recognizeFisher(frame, detectedFaces, recognizer, faceNames)
        elif(self.__name == "eigen"):
            return self.__recognizeEigen(frame, detectedFaces, recognizer, faceNames)

    def __recognizeLbph(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            #face = frame[y : y + w, x : x + h]
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            face = cv2.resize(face, (1000, 1000))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(face, face)
            faceIndex, confidence = recognizer.predict(face)
            print(faceIndex)
            print(confidence)
            if(faceIndex != -1):
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)
        return frame 

    def __recognizeFisher(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            #face = frame[y : y + w, x : x + h]
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            face = cv2.resize(face, (1000, 1000))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(face, face)
            faceIndex, confidence = recognizer.predict(face)
            print(faceIndex)
            print(confidence)
            if(faceIndex != -1):
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)
        return frame
                
    def __recognizeEigen(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            #face = frame[y : y + w, x : x + h]
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            face = cv2.resize(face, (1000, 1000))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(face, face)
            faceIndex, confidence = recognizer.predict(face)
            print(faceIndex)
            print(confidence)
            if(faceIndex != -1):
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)
        return frame
