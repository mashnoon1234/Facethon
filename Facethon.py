import cv2, os
import sys, json        # For command-line arguments
from Video              import Video # The custom Video class
from FaceDetect         import FaceDetect # The custom FaceDetect class
from MachineLearning    import MachineLearning
from FaceRecognize      import FaceRecognize
from pprint             import pprint

 
def main( config ): # Main function
    #video = Video("rtsp://admin:hik12345@192.168.1.21/video.h264")
    #pprint(config)
    weights, gpu = "", 0
    xmlOrCfg     = ""
    videoInput   = config["Video"]["input"]
    frameWidth   = int( config["Video"]["frame_width"] )
    modelName    = config["Detection"]["Algorithm"]
    recognitionAlgo = config["Recognition"]["Algorithm"]
    
    if modelName == "yolo2":
        gpu      = float( config["Detection"]["yolo2"]["gpu"] )
        xmlOrCfg = config["Detection"]["yolo2"]["model_path"]
        weights  = config["Detection"]["yolo2"]["trained_weight_path"]
    
    if( videoInput == "webcam" ):
        # video = Video( 0, frameWidth )               # Video input taken from webcam
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        video = Video( videoInput, frameWidth )      # Video input taken from IP Camera
    try:
        detector = FaceDetect( modelName, xmlOrCfg, weights, gpu)
    except:
        detector = FaceDetect( modelName, xmlOrCfg )

    trainer     = MachineLearning(recognitionAlgo, modelName)
    recognizer  = FaceRecognize(recognitionAlgo)

    trainedRecognizer, faceNames = trainer.trainRecognizer(config["FaceDatabase"]["path"])

    if( videoInput == "webcam"):
        video = Video(0, frameWidth)
    else:
        video = Video(config["Video"]["url"], frameWidth)
    
    while( True ):
        video.startTimer()
        # frame = video.captureFrame()
        ret, frame = capture.read( )
        # frame = video.processFrame(frame)
        frame, detectedFaces = detector.detect(frame, "realtime")
        frame = recognizer.recognize(frame, detectedFaces, trainedRecognizer, faceNames)

        video.stopTimer()
        video.showFrame(frame)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main( json.load(open('config.json')) )
