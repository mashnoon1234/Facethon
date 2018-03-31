import numpy
import cv2
import os

class MachineLearning: # This class contains training and testing algorithms of machine-learning algorithms encapsulated in functions
    def __init__(self, name, detector): # Constructor / Initializer
        self.__name = name
        self.__detector = detector
        self.__lightCorrection = cv2.createCLAHE(60, (3, 3))
    
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
        self.__recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8)
        faces = []
        faceNames = []
        faceIndex = []
        labels = os.listdir(imageDirectory)
        i = 0
        for eachLabel in labels:
            if eachLabel.startswith("."):
                continue
            print(eachLabel)
            imageLabels = os.listdir(imageDirectory + "/" + eachLabel + "/")
            faceNames.append(eachLabel)
            for eachImageLabel in imageLabels:
                if eachImageLabel.startswith("."):
                    continue
                print(eachImageLabel)
                image = cv2.imread(imageDirectory + "/" + eachLabel + "/" + eachImageLabel)
                image, detectedFaces = self.__detector.detect(image, "image")
                (x, y, w, h) = detectedFaces[0]
                face = image[y : y + w, x : x + h]
                if face is not None:
                    face = cv2.resize(face, (800, 800))
                    #b, g, r = cv2.split(face)
                    #self.__lightCorrection.apply(b, b)
                    #self.__lightCorrection.apply(g, g)
                    #self.__lightCorrection.apply(r, r)
                    #face = cv2.merge((b, g, r))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    self.__lightCorrection.apply(face, face)
                    #cv2.equalizeHist(face, face)
                    #face = cv2.GaussianBlur(face, (3, 3), 0)
                    faces.append(face)
                    faceIndex.append(i)
            i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainFisher(self, imageDirectory):
        self.__recognizer = cv2.face.FisherFaceRecognizer_create()
        faces = []
        faceNames = []
        faceIndex = []
        labels = os.listdir(imageDirectory)
        i = 0
        for eachLabel in labels:
            if eachLabel.startswith("."):
                continue
            print(eachLabel)
            imageLabels = os.listdir(imageDirectory + "/" + eachLabel + "/")
            faceNames.append(eachLabel)
            for eachImageLabel in imageLabels:
                if eachImageLabel.startswith("."):
                    continue
                print(eachImageLabel)
                image = cv2.imread(imageDirectory + "/" + eachLabel + "/" + eachImageLabel)
                image, detectedFaces = self.__detector.detect(image, "image")
                (x, y, w, h) = detectedFaces[0]
                face = image[y : y + w, x : x + h]
                if face is not None:
                    face = cv2.resize(face, (800, 800))
                    #b, g, r = cv2.split(face)
                    #self.__lightCorrection.apply(b, b)
                    #self.__lightCorrection.apply(g, g)
                    #self.__lightCorrection.apply(r, r)
                    #face = cv2.merge((b, g, r))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    self.__lightCorrection.apply(face, face)
                    #cv2.equalizeHist(face, face)
                    #face = cv2.GaussianBlur(face, (3, 3), 0)
                    faces.append(face)
                    faceIndex.append(i)
                    #cv2.imshow(eachImageLabel, face)
            i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainEigen(self, imageDirectory):
        self.__recognizer = cv2.face.EigenFaceRecognizer_create()
        faces = []
        faceNames = []
        faceIndex = []
        labels = os.listdir(imageDirectory)
        i = 0
        for eachLabel in labels:
            if eachLabel.startswith("."):
                continue
            print(eachLabel)
            imageLabels = os.listdir(imageDirectory + "/" + eachLabel + "/")
            faceNames.append(eachLabel)
            for eachImageLabel in imageLabels:
                if eachImageLabel.startswith("."):
                    continue
                print(eachImageLabel)
                image = cv2.imread(imageDirectory + "/" + eachLabel + "/" + eachImageLabel)
                image, detectedFaces = self.__detector.detect(image, "image")
                (x, y, w, h) = detectedFaces[0]
                face = image[y : y + w, x : x + h]
                if face is not None:
                    face = cv2.resize(face, (800, 800))
                    #b, g, r = cv2.split(face)
                    #self.__lightCorrection.apply(b, b)
                    #self.__lightCorrection.apply(g, g)
                    #self.__lightCorrection.apply(r, r)
                    #face = cv2.merge((b, g, r))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    self.__lightCorrection.apply(face, face)
                    #cv2.equalizeHist(face, face)
                    #face = cv2.GaussianBlur(face, (3, 3), 0)
                    faces.append(face)
                    faceIndex.append(i)
            i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainYolo2(self): # Trains Yolo2
        pass

    def __testYolo2(self): # Tests Yolo2
        pass
