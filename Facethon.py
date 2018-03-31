import cv2
import sys # For command-line arguments
from Video import Video # The custom Video class
from FaceDetect import FaceDetect # The custom FaceDetect class
from MachineLearning import MachineLearning
from FaceRecognize import FaceRecognize

def main(argv): # Main function
    try:
        detector = FaceDetect(argv[3], argv[4], argv[5], argv[6])
    except:
        detector = FaceDetect(argv[3], argv[4])
    trainer = MachineLearning("fisher", detector)
    recognizer = FaceRecognize("fisher")
    trainedRecognizer, faceNames = trainer.trainRecognizer("Faces/")
    if(argv[1] == "webcam"):
        video = Video(0, argv[2])
    else:
        video = Video(argv[1], argv[2])
    while(True):
        video.startTimer()
        frame = video.captureFrame()
        frame = video.processFrame(frame)
        frame, detectedFaces = detector.detect(frame, "realtime")
        frame = recognizer.recognize(frame, detectedFaces, trainedRecognizer, faceNames)
        video.stopTimer()
        video.showFrame(frame)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break

if __name__ == "__main__":
    if(len(sys.argv) < 4 & len(sys.argv) > 7):
        print("Pass 3 or 4 arguments and re-run the program!")
    else:
        main(sys.argv)
