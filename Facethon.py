import cv2
import sys # For command-line arguments
from Video import Video # The custom Video class
from FaceDetect import FaceDetect # The custom FaceDetect class

def main(argv): # Main function
    #video = Video("rtsp://admin:hik12345@192.168.1.21/video.h264")
    if(argv[1] == "webcam"):
        video = Video(0, argv[2])
    else:
        video = Video(argv[1], argv[2])
    try:
        faceDetect = FaceDetect(argv[3], argv[4], argv[5], argv[6])
    except:
        faceDetect = FaceDetect(argv[3], argv[4])
    while(True):
        video.startTimer()
        frame = video.processFrame(video.captureFrame())
        frame = faceDetect.detect(frame)
        video.stopTimer()
        video.showFrame(frame)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break

if __name__ == "__main__":
    if(len(sys.argv) < 4 & len(sys.argv) > 7):
        print("Pass 3 or 4 arguments and re-run the program!")
    else:
        main(sys.argv)
