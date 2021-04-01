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

videos_path1="E:/data/YawDD/YawDD dataset/Mirror-Table1/1/"
#videos_path2="E:/data/YawDD/YawDD dataset/Mirror-Table1/Male_mirror Avi Videos-1/"

videos1=glob.glob(videos_path1+"*.avi")
#videos2=glob.glob(videos_path2+"*.avi")
videos=[videos1]
countimg=3792
framecount=0
countimg2=0

for i in videos:
    for j in i:
        frameeachcount=0
        dis=[]
        cap = cv2.VideoCapture(j)
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        while (cap.isOpened()):
            ret, frame = cap.read()
            framecount+=1
            frameeachcount+=1
            if ret==False:
                break
        
            faces = detector(frame)
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
                landmarks = predictor(frame, face)
        
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
#                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                        
                    if n==66:
                        y_66=y
                        x_66=x
#                        cv2.circle(frame, (x, y_62), 6, (120, 225, 0), -1)
                    if n==62:
                        y_62=y
                        x_62=x
#                        cv2.circle(frame, (x, y_66), 6, (0, 0, 0), -1)
                    if n==36:
                        y_36=y
                        x_36=x
                    if n==39:
                        y_39=y
                        x_39=x
                        
                    countimg+=1
                    countimg2+=1
                    framedeep=frame[y_36-30:y_36+30, x_36-30:x_39+30]
                    framedeep2=frame[y_66-70:y_62+70, x_66-70:x_62+50]
                    framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
                    framedeep2 = cv2.cvtColor(framedeep2, cv2.COLOR_BGR2GRAY)
                    framedeepequ=cv2.equalizeHist(framedeep)
                    framedeepequ2=cv2.equalizeHist(framedeep2)
                    name1 ="%d.bmp"%(countimg)
                    name2 ="%d.bmp"%(countimg2)
                    filename="E:/data/YawDD/YawDD dataset/Mirror-Table1/sequence/"+name1
                    cv2.imwrite(filename, framedeepequ)
                    filename="E:/data/YawDD/YawDD dataset/Mirror-Table1/sequence/"+name2
                    cv2.imwrite(filename, framedeepequ2)
                        
                        


#                        print(max(dis))
                            
#                        print(y_41-y_37)
#                        if abs(y_37-y_41)> 7*(max(dis)/8):
#                            if framecount%2==0:
#                                countimg+=1
#                                framedeep=frame[y_36-30:y_36+30, x_36-30:x_39+30]
#                                framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
#                                framedeepequ=cv2.equalizeHist(framedeep)
#                                name ="%d.bmp"%(countimg)
#                                filename="E:/data/YawDD/YawDD dataset/Mirror-Table1/Try-Open/"+name
#                                cv2.imwrite(filename, framedeepequ)
                            
                  
                        
#                        
#                    
#

                            
                                    
#                        if 11<abs(y_62 - y_66)<17:
#                            if framecount%3==0:
#                                    countimg+=1
#                                    framedeep=frame[y_66-45:y_66+30, x-30:x+30]
#                                    framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
#                                    framedeepequ=cv2.equalizeHist(framedeep)
#                                    name ="%d.bmp"%(countimg)
#                                    filename="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-talking/"+name
#                                    cv2.imwrite(filename, framedeep)
#                                    
#                                    
#                        if abs(y_62 - y_66)>19:
#                            if framecount%3==0:
#                                    countimg+=1
#                                    framedeep=frame[y_66-45:y_66+30, x-30:x+30]
#                                    framedeep = cv2.cvtColor(framedeep, cv2.COLOR_BGR2GRAY)
#                                    framedeepequ=cv2.equalizeHist(framedeep)
#                                    name ="%d.bmp"%(countimg)
#                                    filename="C:/Users/MaryamHashemi/Desktop/Review/Codes/sequence images-yawing/"+name
#                                    cv2.imwrite(filename, framedeep)
                   
                   

            

        
            
        
        
