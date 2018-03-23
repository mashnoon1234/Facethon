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
        elif(self.__name == "fisher"):
            return self.__trainFisher(imageDirectory)
        elif(self.__name == "eigen"):
            return self.__trainEigen(imageDirectory)

    def trainDetector(self):
        pass
    
    def __trainHaarcascade(self): # Trains Haarcascade Classifiers
        pass

    def __testHaarcascade(self): # Tests Haarcascade Classifiers
        pass
    
    def __trainLBPH(self, imageDirectory):
        self.__recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 55)
        faces = []
        faceNames = []
        faceIndex = []
        imageLabels = os.listdir(imageDirectory)
        #print(imageLabels)
        i = 0
        for eachImageLabel in imageLabels:
            if eachImageLabel.startswith("."):
                continue
            print(eachImageLabel)
            image = cv2.imread(imageDirectory + "/" + eachImageLabel)
            #image = cv2.UMat(image)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detectedFaces = self.__detector.detectMultiScale(image, 1.1, 8)
            print(type(detectedFaces), detectedFaces) # To test if face is empty or not
            (x, y, w, h) = detectedFaces[0]
            face = image[y : y + w, x : x + h]
            #face = cv2.UMat(image, [y, y + w], [x, x + h])
            if face is not None:
                face = cv2.resize(face, (500, 500))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                #cv2.equalizeHist(face, face)
                faces.append(face)
                faceNames.append(eachImageLabel)
                faceIndex.append(i)
                i += 1
    #cv2.imshow(eachImageLabel,face)
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainFisher(self, imageDirectory):
        self.__recognizer = cv2.face.FisherFaceRecognizer_create(0, 50)
        faces = []
        faceNames = []
        faceIndex = []
        imageLabels = os.listdir(imageDirectory)
        #print(imageLabels)
        i = 0
        for eachImageLabel in imageLabels:
            if eachImageLabel.startswith("."):
                continue
            print(eachImageLabel)
            image = cv2.imread(imageDirectory + "/" + eachImageLabel)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detectedFaces = self.__detector.detectMultiScale(image, 1.1, 8)
            print(type(detectedFaces), detectedFaces) # To test if face is empty or not
            (x, y, w, h) = detectedFaces[0]
            face = image[y : y + w, x : x + h]
            if face is not None:
                faces.append(face)
                faceNames.append(eachImageLabel)
                faceIndex.append(i)
                i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainEigen(self, imageDirectory):
        self.__recognizer = cv2.face.EigenFaceRecognizer_create(0, 50)
        faces = []
        faceNames = []
        faceIndex = []
        imageLabels = os.listdir(imageDirectory)
        #print(imageLabels)
        i = 0
        for eachImageLabel in imageLabels:
            if eachImageLabel.startswith("."):
                continue
            print(eachImageLabel)
            image = cv2.imread(imageDirectory + "/" + eachImageLabel)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detectedFaces = self.__detector.detectMultiScale(image, 1.1, 8)
            print(type(detectedFaces), detectedFaces) # To test if face is empty or not
            (x, y, w, h) = detectedFaces[0]
            face = image[y : y + w, x : x + h]
            if face is not None:
                faces.append(face)
                faceNames.append(eachImageLabel)
                faceIndex.append(i)
                i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainYolo2(self): # Trains Yolo2s
        pass

    def __testYolo2(self): # Tests Yolo2s
        pass
