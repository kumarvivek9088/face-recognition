import cv2
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

path=r'D:/FaceData/Vivek'
width=130
height=100  #saved image size = 130 x 100
face_cascade = cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)

count=0
while count < 50:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    for (p,q,r,s) in faces:
        cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
        face=gray[q:q+s,p:p+r]
        face_resize=cv2.resize(face,(width,height))
        cv2.imwrite('%s/%s.png' % (path,count),face_resize)
    count+=1

cv2.imshow('Saving DATA',im)
cv2.imshow('Crop view',face)
cv2.waitKey(10)
    

