import numpy

class MachineLearning: # This class contains training and testing algorithms of machine-learning algorithms encapsulated in functions
    def __init__(self): # Constructor / Initializer
        pass

    def trainHaarcascade(self): # Trains Haarcascade Classifiers
        pass

    def testHaarcascade(self): # Tests Haarcascade Classifiers
        pass
    
    def trainLBPH(self, images, recognizer, imageLabels):
        recognizer.train(images, numpy.array(imageLabels))
        return recognizer

    def trainYolo2(self): # Trains Yolo2s
        pass

def testYolo2(self): # Tests Yolo2s
        pass
