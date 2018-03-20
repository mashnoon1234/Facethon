import numpy
import cv2
import os
from FaceDetect import FaceDetect

class MachineLearning: # This class contains training and testing algorithms of machine-learning algorithms encapsulated in functions
    def __init__(self, name): # Constructor / Initializer
        self.__name = name
        self.__detector = cv2.CascadeClassifier("Pretrained_Models/haarcascade_frontalface_default.xml")
    
    def trainRecognizer(self, imageDirectory):
        if(self.__name == "lbph"):
            return self.__trainLBPH(imageDirectory)

    def trainDetector(self):
        pass
    
    def __trainHaarcascade(self): # Trains Haarcascade Classifiers
        pass

    def __testHaarcascade(self): # Tests Haarcascade Classifiers
        pass
    
    def __trainLBPH(self, imageDirectory):
        self.__recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        faceNames = []
        faceIndex = []
        #imageLabels = os.listdir(imageDirectory)
            #for eachImageLabel in imageLabels:
            #if eachImageLabel.startswith("."):
            #    continue
        image = cv2.imread("Faces/Mashnoon.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectedFaces = self.__detector.detectMultiScale(image, 1.1, 4)
        print(type(detectedFaces), detectedFaces) # To test if face is empty or not
        (x, y, w, h) = detectedFaces[0]
        face = image[y : y + w, x : x + h]
        if face is not None:
            faces.append(face)
            faceNames.append("Mashnoon")
            faceIndex.append(0)
        
        image = cv2.imread("Faces/Mashnoon_1.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectedFaces = self.__detector.detectMultiScale(image, 1.1, 4)
        print(type(detectedFaces), detectedFaces) # To test if face is empty or not
        (x, y, w, h) = detectedFaces[0]
        face = image[y : y + w, x : x + h]
        if face is not None:
            faces.append(face)
            faceNames.append("Mashnoon")
            faceIndex.append(1)
        
        image = cv2.imread("Faces/MLESIR.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectedFaces = self.__detector.detectMultiScale(image, 1.1, 4)
        print(type(detectedFaces), detectedFaces) # To test if face is empty or not
        (x, y, w, h) = detectedFaces[0]
        face = image[y : y + w, x : x + h]
        if face is not None:
            faces.append(face)
            faceNames.append("Lutfe Elahi Sir")
            faceIndex.append(2)
        
        image = cv2.imread("Faces/MLESIR1.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detectedFaces = self.__detector.detectMultiScale(image, 1.1, 4)
        print(type(detectedFaces), detectedFaces) # To test if face is empty or not
        (x, y, w, h) = detectedFaces[0]
        face = image[y : y + w, x : x + h]
        if face is not None:
            faces.append(face)
            faceNames.append("Lutfe Elahi Sir")
            faceIndex.append(3)
        
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames


    def __trainYolo2(self): # Trains Yolo2s
        pass

    def __testYolo2(self): # Tests Yolo2s
        pass
