import os
import cv2
import numpy as np
data_faces = []
label = []
faceCascade = cv2.CascadeClassifier("./haarcascade.xml")

for i in os.listdir(os.path.join(os.getcwd(),"train")):
    img_read = cv2.imread(os.path.join(os.path.join(os.getcwd(),"train"),i))
    gray = cv2.cvtColor(img_read,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+w,x:x+h]
        data_faces.append(roi_gray)
        label.append(1)



face_rec = cv2.face.LBPHFaceRecognizer_create()
face_rec.train(data_faces,np.array(label))

vid = cv2.VideoCapture(0)
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))
    for (x,y,w,h) in faces:
        roi_test_image = gray[y:y+w,x:x+h]
        label,confidence = face_rec.predict(roi_test_image)
        if(label and confidence > 90):
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=5)
            cv2.putText(frame,"Akshay Kumar",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    
    cv2.imshow("Frame", frame)
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 