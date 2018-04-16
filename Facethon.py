import cv2, os
import sys, json # For command-line arguments
from Video import Video # The custom Video class
from FaceDetect import FaceDetect # The custom FaceDetect class
from pprint import pprint

 
def main( config ): # Main function
    #video = Video("rtsp://admin:hik12345@192.168.1.21/video.h264")
    pprint(config)
    weights, gpu = "", 0
    xmlOrCfg     = ""
    videoInput   = config["Video"]["input"]
    frameWidth   = int( config["Video"]["frame_width"] )
    modelName    = config["Detection"]["Algorithm"]
    
    capture      = None

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
        faceDetect = FaceDetect( modelName, xmlOrCfg, weights, gpu)
    except:
        faceDetect = FaceDetect( modelName, xmlOrCfg )
    
    

    while(True):
        # video.startTimer()
        # frame = video.processFrame( video.captureFrame( ) )
        ret, frame = capture.read( )
        frame      = faceDetect.detect( frame )
        # video.stopTimer( )
        # video.showFrame( frame )
        cv2.imshow('Facethon', frame)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break


    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main( json.load(open('config.json')) )