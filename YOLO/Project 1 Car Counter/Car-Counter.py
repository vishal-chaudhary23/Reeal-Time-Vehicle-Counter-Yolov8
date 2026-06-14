from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort.sort import Sort
import numpy as np
import time


capture = cv.VideoCapture('Project 1 Car Counter/Videos/Cars 5.mp4')
# capture.set(3, 400)
# capture.set(4, 600)
pTime =0

model = YOLO('../Yolo-Weights/yolov8n.pt')


mask = cv.imread('Images/ImgRegion.png')

tracker = Sort(max_age=40)

limits = [450,420,1150,420]

totalcount = []

while True:
    success, frame = capture.read()
    # frame = cv.resize(frame, (420, 700))
    frame = cv.resize(frame, (1280,720))
    frameRegion = cv.bitwise_and(frame, mask)

    results = model(frameRegion)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(frame, (x1,y1),(x2,y2), (0,255,0),3)
           
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)

            # Confidence Value 
            conf = math.ceil((box.conf[0] *100))/100
            # conf = box.conf[0]

            # # Class Names
            cls_id = int(box.cls[0])  # class ID (tensor -> int)
            class_name = model.names[cls_id]  # get class name



            if (class_name == 'car' or class_name == 'bus' or class_name == 'truck'\
                  or class_name == 'motorcycle') and conf>0.3:
                # cvzone.cornerRect(frame, bbox)
                # cvzone.putTextRect(frame, f'{conf} {class_name}',(x1, y1), scale= 1, offset= 3, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        cvzone.cornerRect(frame, (x1, y1, w, h), colorR= (255,0,0), rt= 3)
        cvzone.putTextRect(frame, f'{int(id)}',(x1, y1), scale= 2, offset= 8, thickness=3)

        cx, cy = x1+w//2, y1+h//2
        cv.circle(frame, (cx,cy), 5, (255,0,255), -1)
        
        if limits[0]<cx< limits[2] and limits[1]-20<cy< limits[1]+20:
            if totalcount.count(id) ==0:
                totalcount.append(id)


    cvzone.putTextRect(frame, f'{len(totalcount)}', (50,50))
    cv.line(frame, (450, 420), (1150, 420), (0,0,255), 3)


    cTime = time.time()

    fps_ms = (cTime - pTime) * 1000  # milliseconds per frame

    # Put the ms/frame on the image
    cv.putText(frame, f"{fps_ms:.2f} ms/frame", (1000, 40),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv.putText(frame, f'FPS: {int(fps)}', (1050, 70),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv.imshow('Video',frame)
    # cv.imshow('Img',ImgRegion)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
