from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

#cap = cv2.VideoCapture("ped_video.mp4")
rtsp_url = "rtsp://labstudent:Erclab_717@10.39.8.182:554/vc=1"
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()
    
model = YOLO("yolov8n.pt")
classNames = ["Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet", "TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"]

mask = cv2.imread("1280map.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [155,549,717,549]
totalCount = []

while True:
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img, mask)
    imgRegion = img
    results = model(imgRegion, stream=True) #uses generator
    detections = np.empty((0,5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            #cvzone.cornerRect(imgRegion, (x1, y1, w, h),l=9)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
           #cvzone.putTextRect(imgRegion, f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.8,thickness=2) # CLASS AND CONF
            currentArray = np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections, currentArray))
            
    
    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1,y1,w,h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(id)}',(max(0,x1),max(35,y1)),scale=0.8,thickness=2) # TRACKNG ID
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        if limits[0]<cx<limits[2] and limits[1] -20 < cy < limits[1] + 20:
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
        
    cvzone.putTextRect(img,f' Count: {len(totalCount)}',(50,50))    

    cv2.imshow("Image", img)
    #cv2.imshow("Image", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
