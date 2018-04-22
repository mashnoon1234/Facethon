import numpy 
import cv2
import os
import face_recognition

class MachineLearning:                      # This class contains training and testing algorithms of machine-learning algorithms encapsulated in functions
    def __init__(self, name, detector):     # Constructor / Initializer
        self.__name             = name
        self.__detector         = detector
        self.__lightCorrection  = cv2.createCLAHE(80, (4, 4))
    
    def trainRecognizer(self, imageDirectory):
        if(self.__name == "lbph"):
            return self.__trainLBPH(imageDirectory)
        elif(self.__name == "fisher"):
            return self.__trainFisher(imageDirectory)
        elif(self.__name == "eigen"):
            return self.__trainEigen(imageDirectory)
        elif(self.__name == "dlib-svm"):
            self.__face_database = {
                'face_names'    : [],
                'face_encodings': []
            }
            self.__trainSVM( imageDirectory )
            self.colors = [ tuple(255 * numpy.random.rand(3)) for _ in range(10) ]
            return self, []

    def trainDetector(self):
        pass
    
    def __trainHaarcascade(self): # Trains Haarcascade Classifiers
        pass

    def __testHaarcascade(self): # Tests Haarcascade Classifiers
        pass
    

    # This function generated landmarks for existing faces from face_database

    def __trainSVM( self, imageDirectory ):
        for img in os.scandir(imageDirectory):
            image               = face_recognition.load_image_file(img.path)    # later we will take it from command line argument
            image_face_encoding = face_recognition.face_encodings(image)[0]
            image_name          = img.name.split('.')[0]
            image_name          = image_name.split('_')[0]
            self.__face_database[ 'face_names' ].append( image_name )
            self.__face_database[ 'face_encodings' ].append( image_face_encoding )
            
    # This Function uses SVM classifier to get prediction for detected faces
    def getPredictedNames( self, predictedFaceEncodings ):
        face_names = []
        for single_face_endcoding in predictedFaceEncodings:
            matches = face_recognition.compare_faces(
                      self.__face_database['face_encodings'], single_face_endcoding )
            name    = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name        = self.__face_database['face_names'][ match_index ]
            face_names.append( name )
            
        return face_names

    def __trainLBPH(self, imageDirectory):
        self.__recognizer = cv2.face.LBPHFaceRecognizer_create(radius = 1, neighbours = 4, grid_x = 8, grid_y = 8, threshold = 2000) #dummy value of threshold
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
                    face = cv2.resize(face, (20, 20))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    self.__lightCorrection.apply(face, face)
                    faces.append(face)
                    faceIndex.append(i)
            i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainFisher(self, imageDirectory):
        self.__recognizer = cv2.face.FisherFaceRecognizer_create(num_components = 0, threshold = 500)
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
                    face = cv2.resize(face, (400, 400))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    self.__lightCorrection.apply(face, face)
                    faces.append(face)
                    faceIndex.append(i)
                    cv2.imshow(eachImageLabel, face)
            i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainEigen(self, imageDirectory):
        self.__recognizer = cv2.face.EigenFaceRecognizer_create(num_components = 0, threshold = 1400)
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
                    face = cv2.resize(face, (20, 20))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    self.__lightCorrection.apply(face, face)
                    faces.append(face)
                    faceIndex.append(i)
            i += 1
        self.__recognizer.train(faces, numpy.array(faceIndex))
        return self.__recognizer, faceNames

    def __trainYolo2(self): # Trains Yolo2
        pass

    def __testYolo2(self): # Tests Yolo2ss
        pass
