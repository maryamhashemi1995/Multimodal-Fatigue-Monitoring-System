# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:09:36 2020

@author: MaryamHashemi
"""

import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import glob

videos_path1="E:/data/YawDD/YawDD dataset/Dash-Table2/Male/"

videos1=glob.glob(videos_path1+"*.avi")
videos=[videos1]
countimg=3000
for i in videos:
    for j in i:
        cap = cv2.VideoCapture(j)
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


        dis1=[]
        dis2=[]
        dis3=[]
        framecount=0
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            faces = detector(gray)
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
                landmarks = predictor(gray, face)
        
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
        #            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                        
                    if n==62:
                        y_62=y
#                        cv2.circle(frame, (x, y_62), 6, (120, 225, 0), -1)
                    if n==66:
                        y_66=y
#                        cv2.circle(frame, (x, y_66), 6, (0, 0, 0), -1)
                        
                        if abs(y_62 - y_66)<10:
                            if framecount%3==0:
                                    countimg+=1
                                    framedeep=frame[y_66-45:y_66+30, x-30:x+30]
                                    framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
                                    framedeepequ=cv2.equalizeHist(framedeep)
                                    name ="%d.jpg"%(countimg)
                                    filename="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-close/"+name
                                    cv2.imwrite(filename, framedeep)
                                    
                        if 10<abs(y_62 - y_66)<20:
                            if framecount%3==0:
                                    countimg+=1
                                    framedeep=frame[y_66-45:y_66+30, x-30:x+30]
                                    framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
                                    framedeepequ=cv2.equalizeHist(framedeep)
                                    name ="%d.jpg"%(countimg)
                                    filename="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-talking/"+name
                                    cv2.imwrite(filename, framedeep)
                                    
                                    
                        if abs(y_62 - y_66)>20:
                            if framecount%3==0:
                                    countimg+=1
                                    framedeep=frame[y_66-45:y_66+30, x-30:x+30]
                                    framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
                                    framedeepequ=cv2.equalizeHist(framedeep)
                                    name ="%d.jpg"%(countimg)
                                    filename="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-yawing/"+name
                                    cv2.imwrite(filename, framedeep)
                   
                   
        
                        
                    
            dis2.append(abs(y_62-y_66))

            framecount=framecount+1
            cv2.imshow("Frame", frame)
            
            key = cv2.waitKey(1)
            if key == 27 or len(dis2)>1000: 
                cv2.destroyAllWindows()
                break
        
        cv2.destroyAllWindows()
        

        
            
        
        
