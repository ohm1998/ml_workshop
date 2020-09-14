import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

vid = cv2.VideoCapture(0)

while True:
    ret , frame = vid.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    print(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
    cv2.imshow("image",frame)
    if cv2.waitKey(1)==13:
        break
vid.release()
cv2.destroyAllWindows()


