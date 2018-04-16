import cv2
import numpy as np
from FaceRecognize import FaceRecognize
from Yolov2.darkflow.net.build import TFNet


class FaceDetect: # This class contains all detection algorithms encapsulated in functions
    def __init__(self, name, xmlOrCfg, weights = "", gpu = 0): # Constructor / Initializer
        self.__name = name
        if(self.__name == "haar"):
            self.__cascade = cv2.CascadeClassifier(xmlOrCfg)
        elif(self.__name == "yolo2"):
            self.__cfg     = xmlOrCfg
            self.__weights = weights
            self.__gpu     = gpu 
            options        = {
                'model': str(self.__cfg),
                'load' : str(self.__weights),
                'threshold': 0.3,
                'gpu': self.__gpu
            }
            print("Options ----> " + str(options))
            self.__tfnet         = TFNet( options )
            self.__faceRecognize = FaceRecognize( "svm" )

    def detect(self, frame): # This function reads the command-line arguments and decides which algorithm to use
        if(self.__name == "haar"):
            return self.__detectHaarcascade(frame)
        elif(self.__name == "yolo2"):
            return self.__detectYolo2( frame )

    def __detectHaarcascade(self, frame): # Haarcascade Face Detection
        detectedFaces = self.__cascade.detectMultiScale(frame, 1.3, 4) # Receives detected faces as an object
        for (x, y, w, h) in detectedFaces: # Iterates through the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Draws rectangles around detected faces
        return frame

    def __predicted_face_locations( self, results ):
        locations = []
        for result in results:
            curr_face = ( result['topleft']['y'], result['bottomright']['x'], 
                          result['bottomright']['y'], result['topleft']['x'] )
            locations.append( curr_face )

        return locations

    def __detectYolo2(self, frame): # Yolo2 Face Detection
        result          = self.__tfnet.return_predict( frame )
        face_locations  = self.__predicted_face_locations( result )
        frame           = self.__faceRecognize.recognize( frame, face_locations )
        return frame
