import cv2

class FaceRecognize: # This class contains all recognition algorithms encapsulated in functions
    def __init__(self, name, xml): Constructor / Initializer
        self.__name = name
        if(self.__name == "lbph"):
            self.__recognizer = cv2.face.createLBPHFaceRecognizer()

    def recognize(self): # This function reads the command-line arguments and decides which algorithm to use
        if(self.__name == "lbph"):
            return self.__recognizeLbph(frame)

    def __recognizeLbph(self, frame, detectedFaces):
        for eachFace in detectedFaces:
            self.__recognizer.predict(eachFace)
