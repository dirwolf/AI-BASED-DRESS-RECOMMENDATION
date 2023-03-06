import cv2
import subprocess
import os
import pyttsx3
import mediapipe as mp
from playsound import playsound
import numpy as np
import pygame
import time
from time import sleep 
import math
from numpy.lib import utils
from tkinter import *

def fcall(a):
    sleep(5)
    mpPose = mp.solutions.pose
    mpFaceMesh = mp.solutions.face_mesh
    facemesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
    mpDraw = mp.solutions.drawing_utils
    drawing = mpDraw.DrawingSpec(thickness = 1 , circle_radius = 1)
    pose = mpPose.Pose()
    capture = cv2.VideoCapture(0)
    lst=[]
    n=0
    scale = 3
    ptime = 0
    count = 0
    brake = 0
    x=150
    y=195
    def speak(audio):

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('rate',150)

        engine.setProperty('voice', voices[0].id)
        engine.say(audio)

        # Blocks while processing all the currently
        # queued commands
        engine.runAndWait()
    speak("We are about to measure your height")
    speak("sir or mam please stay still")
    # speak("Although I reach a precision upto ninety eight percent")
    count=0
    while True:
        isTrue,img = capture.read()
        img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for id,lm in enumerate(result.pose_landmarks.landmark):
                lst[n] = lst.append([id,lm.x,lm.y])
                n+1
                # print(lm.z)
                # if len(lst)!=0:
                #     print(lst[3])
                h , w , c = img.shape
                if id == 32 or id==31 :
                    cx1 , cy1 = int(lm.x*w) , int(lm.y*h)
                    cv2.circle(img,(cx1,cy1),15,(0,0,0),cv2.FILLED)
                    d = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
                    # height = round(utils.findDis((cx1,cy1//scale,cx2,cy2//scale)/10),1)
                    di = round(d*0.5)
                    # pygame.mixer.init()
                    # pygame.mixer.music.load("check.mp3")
                    # pygame.mixer.music.play()
                    # speak(f"You are {di} centimeters tall")
                    # speak("I am done")
                    # speak("You can relax now")
                    # speak("Press q and give me some rest now.")
                    # if ord('q'):
                    #     cv.destroyAllWindows()
                    #     break

                    dom = ((lm.z-0)**2 + (lm.y-0)**2)**0.5
                    # height = round(utils.findDis((cx1,cy1//scale,cx2,cy2//scale)/10),1)
                    cv2.putText(img ,"Height : ",(40,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),thickness=2)
                    cv2.putText(img ,str(di),(180,70),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),thickness=2)
                    cv2.putText(img ,"cms" ,(240,70),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),thickness=2)
                    cv2.putText(img ,"Stand atleast 3 meter away" ,(40,450),cv2.FONT_HERSHEY_PLAIN,2,	(0,0,255),thickness=2)

                    count+=1
                    if(count==300):
                        print(di)
                        result=""
                        if(di>140 and di<150):
                            result="dress 1"
                        elif(di>150 and di<160):
                            result="dress 2"  
                        elif(di>160 and di<170):
                            result="dress 3" 
                        elif(di>170 and di<180):
                            result="dress 4"   
                        elif(di>180 and di<190):
                            result="dress 5"  
                        print(a,result)     
                        cv2.destroyAllWindows()  

                        window=Tk()
                        window.title("result")
                        p1 = PhotoImage(file = 'savicon.png')
                        
                        window.iconphoto(False, p1)
                        a=a+'\n'+result
                        Label(window,text=a,font=("Times New Roman",25),width=10).pack(pady=10)
                        Button(window,text="Shop Now",font=("Consolas",25),width=10,bg='#15ff00').pack(pady=10,side='left')
                        Button(window,text="Dress location",font=("Consolas",25),width=15,bg='#15ff00').pack(pady=10,side='right')
                        window.resizable(False,False)
                        window.mainloop() 
                        break            
                        
                
                    # cv.putText(img ,"Go back" ,(240,70),cv.FONT_HERSHEY_PLAIN,2,(255,255,0),thickness=2)
                if id == 6:
                    cx2 , cy2 = int(lm.x*w) , int(lm.y*h)
                    # cx2 = cx230
                    cy2 = cy2 + 20
                    cv2.circle(img,(cx2,cy2),15,(0,0,0),cv2.FILLED)
        img = cv2.resize(img , (700,500))
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img , "FPS : ",(40,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),thickness=2)
        cv2.putText(img , str(int(fps)),(160,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),thickness=2)
        cv2.imshow("Task",img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def mcall(a):
    cmd = 'Body_Detection.py'
    # distance from camera to object(face) measured
    # centimeter
    Known_distance = 60.96

    # width of face in the real world or Object Plane
    # centimeter
    Known_width = 14.3

    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    def speak(audio):

        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('rate',150)

        engine.setProperty('voice', voices[0].id)
        engine.say(audio)

        # Blocks while processing all the currently
        # queued commands
        engine.runAndWait()

    # defining the fonts
    fonts = cv2.FONT_HERSHEY_COMPLEX

    # face detector object
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # focal length finder function
    def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):

        # finding the focal length
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    # distance estimation function
    def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):

        distance = (real_face_width * Focal_Length)/face_width_in_frame

        # return the distance
        return distance


    def face_data(image):

        face_width = 0 # making face width to zero

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detecting face in the image
        faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

        # looping through the faces detect in the image
        # getting coordinates x, y , width and height
        for (x, y, h, w) in faces:

            # draw the rectangle on the face
            cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)

            # getting face width in the pixels
            face_width = w

        # return the face width in pixel
        return face_width


    # reading reference_image from directory
    ref_image = cv2.imread("Ref_image.jpg")

    # find the face width(pixels) in the reference_image
    ref_image_face_width = face_data(ref_image)

    # get the focal by calling "Focal_Length_Finder"
    # face width in reference(pixels),
    # Known_distance(centimeters),
    # known_width(centimeters)
    Focal_length_found = Focal_Length_Finder(
        Known_distance, Known_width, ref_image_face_width)

    # print(Focal_length_found)

    # show the reference image
    cv2.imshow("ref_image", ref_image)

    # initialize the camera object so that we
    # can get frame from it
    cap = cv2.VideoCapture(0)

    # looping through frame, incoming from
    # camera/video
    while True:

        # reading the frame from camera
        _, frame = cap.read()

        # calling face_data function to find
        # the width of face(pixels) in the frame
        face_width_in_frame = face_data(frame)

        # check if the face is zero then not
        # find the distance
        if face_width_in_frame != 0:
        
            # finding the distance by calling function
            # Distance distance finder function need
            # these arguments the Focal_Length,
            # Known_width(centimeters),
            # and Known_distance(centimeters)
            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame)

            # draw line as background of text
            cv2.line(frame, (30, 30), (230, 30), RED, 32)
            cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
            Distance = round(Distance)
            if Distance in range(100 , 360):
                speak("Stand there and dont move")
                cv2.destroyAllWindows()
                fcall(a)
                break
            elif Distance < 100 :
                speak("Step back")
            else:
                speak("Come a little closer")

            # Drawing Text on the screen
            cv2.putText(
                frame, f"Distance: {round(Distance,2)} cms", (30, 35),
            fonts, 0.6, GREEN, 2)

        # show the frame on the screen
        cv2.imshow("frame", frame)

        # quit the program if you press 'q' on keyboard
        if cv2.waitKey(1) == ord("q"):
            break

    # closing the camera
    cap.release()

    # closing the the windows that are opened
    cv2.destroyAllWindows()

def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]

    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)

    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"



faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


video=cv2.VideoCapture(0)

padding=20
c=0
while True:
    ret,frame=video.read()
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]


        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        c+=1
        if(c==200):
            cv2.destroyAllWindows()
            # os.system('python ex.py')
            mcall(label)
            break
            

    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

