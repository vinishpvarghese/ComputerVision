# Face Recognition using Open CV Library

import cv2
import time
from datetime import datetime

face_cascade = cv2.CascadeClassifier('haar_frontalface.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haar_eye.xml') # We load the cascade for the eyes.
smile_cascade = cv2.CascadeClassifier('haar_smile.xml') # We load the cascade for the eyes.



def writeData(): # We write the date and time of the face detection to a file for Analytics purpose   
    with open('Detection.log', 'a') as file:
        file.write('Face Detection Recorded at: %s\n' %datetime.now())

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    
#    if faces is None:
#        print("No Face detected ")
#    else:
#        writeData()
    for (x, y, w, h) in faces: # For each detected face:
        writeData() # if face detected log the date and time of face detection 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
        
        #smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 3) # We apply the detectMultiScale method to locate one or several eyes in the image.
        #for (sx, sy, sw, sh) in smiles: # For each detected eye:
        #    cv2.rectangle(roi_color,(sx, sy),(sx+sw, sy+sh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
    return frame # We return the image with the detector rectangles.



def getfps():
    # Start default camera
    video = cv2.VideoCapture(0);
     
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
     
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
 
    # Number of frames to capture
    num_frames = 120;
     
    print("Capturing {0} frames".format(num_frames))
 
    # Start time
    start = time.time()
     
    # Grab a few frames
    for i in range(0, num_frames) :
        ret, frame = video.read()
 
     
    # End time
    end = time.time()
 
    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
 
    # Calculate frames per second
    fps  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps));
 
    # Release video
    video.release()
    
    return fps

fps = getfps() # We find the frames per second 
video_capture = cv2.VideoCapture(0) # We turn the webcam on.

#For turning on the IP CAM HIK Vision 
#"rtsp://admin:fuTtJqR7@192.168.0.64:554/Streaming/channels/2/"
#cap = cv2.VideoCapture()
#video_capture.open("rtsp://admin:Aveiro35@169.254.103.173:554/Streaming/channels/2/")


while True: # We repeat infinitely (until break):
    _, frame = video_capture.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.
    cv2.imshow('Video', canvas) # We display the outputs.
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # Turn the webcam off.
cv2.destroyAllWindows() # Destroy all the windows inside which the images were displayed.