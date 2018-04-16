import cv2
import numpy as np
from FaceRecognize import FaceRecognize
from Yolov2.darkflow.net.build import TFNet


class FaceDetect: # This class contains all detection algorithms encapsulated in functions
<<<<<<< HEAD
    def __init__(self, name, xmlOrCfg, weights = "", gpu = 0): # Constructor / Initializer
=======
    def __init__(self, name, xmlOrCfg, weights = None, gpu = 0.0): # Constructor / Initializer
>>>>>>> b9ddca4de8fa1fc5b684c38443e2cdd821cc6365
        self.__name = name
        if(self.__name == "haar"):
            self.__cascade = cv2.CascadeClassifier(xmlOrCfg)
        elif(self.__name == "yolo2"):
            self.__cfg     = xmlOrCfg
            self.__weights = weights
            self.__gpu     = float(gpu) 
            options        = {
                'model': str(self.__cfg),
<<<<<<< HEAD
                'load' : str(self.__weights),
=======
                'load': str(self.__weights), #str(self.__weights),
>>>>>>> b9ddca4de8fa1fc5b684c38443e2cdd821cc6365
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
        frame = cv2.UMat.get(frame)
        result          = self.__tfnet.return_predict( frame ) # Issue here for some reason!
        face_locations  = self.__predicted_face_locations( result )
        frame           = self.__faceRecognize.recognize( frame, face_locations )
        return frame
