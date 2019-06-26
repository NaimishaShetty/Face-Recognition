import numpy as np
import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

person01 = np.load('NShetty.npy').reshape(20,50*50*3)
person02 = np.load('ABC.npy').reshape(20,50*50*3)

names = {
        0:'Naimisha',
        1:'ABC',
        }

data = np.concatenate([person01,person02])
labels = np.zeros((40,1))
labels[:21,:] = 0.0
labels[21:,:] = 1.0


def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(testinput,data,labels,k):
    numRow=data.shape[0]
    dist = []
    for i in range(numRow):
        dist.append(distance(testinput,data[i]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sortedLabels = labels[indx][:k]
    counts = np.unique(sortedLabels,return_counts=True)
    return counts[0][np.argmax(counts[-1])]

#sampleTest = [7,1,8,2,9,3]
#sampleTest = np.asarray(sampleTest).reshape(3,2)
#
#smpleLabel = np.array([1,1,0])
#sampleInput = np.array([3,4]).reshape(1,2)
#
#knn(sampleInput,sampleTest,sampleLabel,3)
    
while True:
    ret, frame = cap.read()
    grayFace = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFace,1.3,5)
    
    for(x,y,w,h) in faces:
        croppedFace = frame[y:y+h,x:x+w,:]
        resizedFace = cv2.resize(croppedFace,(50,50))
        prediction = knn(resizedFace.flatten(),data,labels,5)
        name = names[int(prediction)]
        cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Face Recognition',frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    