#from ultralytics import YOLO 
#import cv2 
#model = YOLO('yolov8n.pt') 
#img   = cv2.imread('traffic.png') 
#results = model(img)
#for box in results.boxes:            
#     class_id = int(box.cls[0])    
#     print(model.names[class_id])

#error is boxes apply to particula item of results

from ultralytics import YOLO 
import cv2 
model = YOLO('yolov8n.pt') 
img   = cv2.imread('traffic.png') 
results = model(img)
for box in results[0].boxes:            
     class_id = int(box.cls[0])    
     print(model.names[class_id])