#importing Modules

from cv2 import cv2
import numpy as np

#Capturing Video through webcam.
cap = cv2.VideoCapture(0)

while(1):
        _, img = cap.read()

        #converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        #defining the range of Yellow color
        # yellow_lower = np.array([22,40,150],np.uint8)
        # yellow_upper = np.array([80,255,255],np.uint8)

        # yellow_lower = np.array([0,100,100],np.uint8)
        # yellow_upper = np.array([5,255,255],np.uint8)

        red_lower = np.array([136, 87, 111],np.uint8)
        red_upper = np.array([180, 255, 255],np.uint8)

        #finding the range yellow colour in the image
        # yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        red = cv2.inRange(hsv, red_lower, red_upper)

        #Morphological transformation, Dilation         
        kernal = np.ones((5 ,5), "uint8")

        # blue=cv2.dilate(yellow, kernal)
        blue=cv2.dilate(red, kernal)
		
        # res=cv2.bitwise_and(img, img, mask = yellow)
        res=cv2.bitwise_and(img, img, mask = red)

        #Tracking Colour (Yellow) 
        (contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>900):
                        
                        x,y,w,h = cv2.boundingRect(contour)     
                        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                        
        cv2.imshow("Color Tracking",img)
        # img = cv2.flip(img,1)
        # cv2.imshow("Yellow",res)
        cv2.imshow("Red",res)
        # cv2.imshow("Yellow2",img)
        # cv2.imshow("blue",blue)
        if cv2.waitKey(10) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                break