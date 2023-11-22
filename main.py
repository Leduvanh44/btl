import cv2
import tracker
import mediapipe as mp
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
import torch
import dlib
import numpy as np
import datetime
skip_frames = 24

ct = CentroidTracker(maxDisappeared=50, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

total = []
move_out = []
move_in =[]
out_time = []
in_time = []

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=False)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    rects = []
    ret, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (H, W) = img.shape[:2]
    if totalFrames % skip_frames == 0:
        trackers = []
        results = model(img)
        for det in results.pred[0]:
                class_id, bbox= det[5], det[:4]
                if class_id == 0:
                    x1, y1, x2, y2 = map(int, bbox)

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    print('trackers:' , trackers)
    else:
        for tracker in trackers:
            tracker.update(rgb)
            pos = tracker.get_position()
            x1 = int(pos.left())
            y1 = int(pos.top())
            x2 = int(pos.right())
            y2 = int(pos.bottom())
            rects.append((x1, y1, x2, y2))

    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    move_out.append(totalUp)
                    out_time.append(date_time)
                    to.counted = True
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    move_in.append(totalDown)
                    in_time.append(date_time)
                    to.counted = True
                    total = []
                    total.append(len(move_in) - len(move_out))
        trackableObjects[objectID] = to      
        text = "ID {}".format(objectID)
        print('text:', text)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.circle(img, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
    cv2.imshow("Real-Time Monitoring/Analysis Window", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    totalFrames += 1
cv2.destroyAllWindows()
		




    # cv2.putText(rgb, f'ID', (x1, y1 - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(1)
