import cv2

img = cv2.imread("./dog.jpg")
vid = cv2.VideoCapture(0)

while True:
    ret,frame = vid.read()
    img_bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_bw,(7,7),0)

    img_edge = cv2.Canny(img_blur,10,70)

    ret,mask =  cv2.threshold(img_edge,70,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold Image",mask)
    if cv2.waitKey(1)==13:
        cv2.imwrite("sketch.jpg",mask)
        break
#cv2.imshow("Blur image",img_blur)
