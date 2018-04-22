import os, cv2
import face_recognition
import numpy as np
import time

class FaceRecognize: # This class contains all recognition algorithms encapsulated in functions
    def __init__(self, name): # Constructor / Initializer
        self.__name = name
        self.__lightCorrection = cv2.createCLAHE(80, (4, 4))

    def recognize(self, frame, detectedFaces, recognizer, faceNames): # This function reads the command-line arguments and decides which algorithm to use
        if(self.__name == "lbph"):
            return self.__recognizeLbph(frame, detectedFaces, recognizer, faceNames)
        elif(self.__name == "fisher"):
            return self.__recognizeFisher(frame, detectedFaces, recognizer, faceNames)
        elif(self.__name == "eigen"):
            return self.__recognizeEigen(frame, detectedFaces, recognizer, faceNames)
                self.__modelName = modelName
        elif self.__name == "svm":
            self.__face_database = {
                'face_name' : [],
                'face_encodings': []
            }
            self.__processExistingFaceDatabase( )
            self.__colors = [ tuple(255 * np.random.rand(3)) for _ in range(10) ]
            frame = self.__recognizeWithSVM( frame, detectedFaces )
            return frame

    def __recognizeLbph(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            face = cv2.resize(face, (20, 20))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            self.__lightCorrection.apply(face, face)
            faceIndex, distance = recognizer.predict(face)
            if(faceIndex != -1):
                print("name : " + faceNames[faceIndex] + ", distance : " + str(distance))
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)
        return frame
    
    def __recognizeFisher(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            face = cv2.resize(face, (20, 20))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            self.__lightCorrection.apply(face, face)
            faceIndex, distance = recognizer.predict(face)
            if(faceIndex != -1):
                print("name : " + faceNames[faceIndex] + ", distance : " + str(distance))
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)
        return frame
                
    def __recognizeEigen(self, frame, detectedFaces, recognizer, faceNames):
        for (x, y, w, h) in detectedFaces:
            face = cv2.UMat(frame, [y, y + w], [x, x + h])
            face = cv2.resize(face, (20, 20))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            self.__lightCorrection.apply(face, face)
            faceIndex, distance = recognizer.predict(face)
            if(faceIndex != -1):
                print("name : " + faceNames[faceIndex] + ", distance : " + str(distance))
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, faceNames[faceIndex], (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)
        return frame

    # This function generated landmarks for existing faces from face_database
    def __processExistingFaceDatabase( self ):
        for img in os.scandir('face_database'):                                 # For now assuming 'face_database' is the folder for existing faces.
            image               = face_recognition.load_image_file(img.path)    # later we will take it from command line argument
            image_face_encoding = face_recognition.face_encodings(image)[0]
            imgae_name          = img.name.split('.')[0]
            self.__face_database[ 'face_names' ].append( imgae_name )
            self.__face_database[ 'face_encodings' ].append( image_face_encoding )
            
    
    # This Function uses SVM classifier to get prediction for detected faces
    def __getPredictedNames( self, predictedFaceEncodings ):
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


    # This is the recognition function which uses SVM classifier.
    def __recognizeWithSVM( self, frame, prectictedFaceLocations ):
        predictedFaceEncodings = face_recognition.face_encodings( frame, prectictedFaceLocations )
        face_names             = self.__getPredictedNames( predictedFaceEncodings )
        
        for color, result, name in zip( self.__colors, prectictedFaceLocations, face_names ):
            tl = ( result[ 3 ], result[ 0 ] )
            br = ( result[ 1 ], result[ 2 ] )

            frame = cv2.rectangle( frame, tl, br, color, 6 )
            frame = cv2.putText(
                frame, name, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2) )

        return frame
