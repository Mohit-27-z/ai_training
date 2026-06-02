#import cv2 
#img = cv2.imread('traffic.jpg')
#cv2.imshow('Traffic', img) 
#cv2.destroyAllWindows()
# you dont add cv.waiKey before detsroyallwindows()
import cv2 
img = cv2.imread('traffic.jpg')
cv2.imshow('Traffic', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()

