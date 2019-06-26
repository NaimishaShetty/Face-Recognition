import numpy as np
import cv2
import matplotlib.pyplot as plt

cap=cv2.VideoCapture(0)
image = cv2.imread('image.jpg',-1)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faceData = []

faceCount = 0

while True:
    
    ret,frame = cap.read()
    
    if ret == True:
        grayFace = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayFace,1.3,5)
        for (x,y,w,h) in faces:
            croppedFace = frame[y:y+h,x:x+w]
            resizedFace = cv2.resize(croppedFace,(50,50))
            #cv2.imwrite('sample'+str(faceCount)+'.jpg',resizedFace)
            faceData.append(resizedFace)
            faceCount+=1
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            
        cv2.imshow('Capturing Frames',frame)
        if cv2.waitKey(1) == 27 or len(faceData) >= 20:
            break
    else:
        print("Camera error")

cap.release()
cv2.destroyAllWindows()

faceData = np.asarray(faceData)

np.save('ABC',faceData)