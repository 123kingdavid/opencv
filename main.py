import cv2


face_hear_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')
image = cv2.imread('63bbecdaa3814ff4e681401d39d00d53.jpg')
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('grey', grey)
cv2.waitKey()

faces = face_hear_cascade.detectMultiScale(grey, 1.1, 4)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0),5)
cv2.imshow('image',image)
cv2.waitKey()