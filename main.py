from ultralytics import YOLO
import cv2
import math
import torch
import numpy as np
from sklearn.cluster import KMeans

def sort_prediction(results):
  """
  From yolo results object class, return sorted prediction array and associated indices
  """
  clusters = KMeans(n_clusters=3).fit_predict(results.boxes.xywhn[:,1].reshape(-1, 1))

  order = np.argsort(clusters)
  clustered = torch.cat((results.boxes.xywhn[order], torch.Tensor(order).unsqueeze(1)),dim=1)
  min_val, max_val = min(clustered[:,1]), max(clustered[:,1])

  groups = [clustered[:3], clustered[3:6], clustered[6:]]

  rows = [None for _ in range(3)]
  for group in groups:
    if abs(group[0,1] - min_val) <= 0.1:
      rows[0] = group
    elif abs(group[0,1] - max_val) <= 0.1:
      rows[2] = group
    else:
      rows[1] = group

  for i, row in enumerate(rows):
    if row is None:
       return None, None
    ids = torch.argsort(row[:,0], descending=False)
    rows[i] = row[ids]

  sorted = torch.cat(tuple(rows), dim=0)
  idxs = sorted[:,4].to(dtype=int)


  return results.boxes.cls[idxs], idxs

def draw_prediction_squares(img, results):
   
   # get top left corner of frame
    x, y = 0, 0

    ordered_classes, _ = sort_prediction(results)
    if ordered_classes is None:
       return
    # create 3x3 grid of squares starting at top left corner
    for i in range(3):
        for j in range(3):
            # get color of square
            color = int(ordered_classes[i*3+j].item())
            # draw square
            cv2.rectangle(img, (x+j*100, y+i*100), (x+j*100+100, y+i*100+100), colors[sticker_classNames[color]], -1)
            # draw border
            cv2.rectangle(img, (x+j*100, y+i*100), (x+j*100+100, y+i*100+100), (0, 0, 0), 2)

model = YOLO('models/cube_detector.pt')
sticker_labeler = YOLO('models/sticker_labeler.pt')
cube_classNames = ["Rubik's Cube"]
sticker_classNames = ['green', 'blue', 'orange', 'red', 'yellow', 'white']

colors = {
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'orange': (0, 165, 255),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255)
}

cap = cv2.VideoCapture(0)
box_window = cv2.namedWindow('detected cube', cv2.WINDOW_NORMAL)

while True:
    ret, img = cap.read()
    # results = model(img)

    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # crop frame to box
            crop = img[y1:y2, x1:x2]

            sticker_results = sticker_labeler(crop, stream=True)
            
            for sticker in sticker_results:
                sticker_boxes = sticker.boxes
                print("labels --->", sticker_boxes.cls)
                if len(sticker_boxes.cls) == 9:
                    draw_prediction_squares(img, sticker)
                for sticker_box in sticker_boxes:
                    _x1, _y1, _x2, _y2 = sticker_box.xyxy[0]
                    _x1, _y1, _x2, _y2 = int(_x1), int(_y1), int(_x2), int(_y2)
                    
                    cv2.rectangle(crop, (_x1, _y1), (_x2, _y2), (255, 0, 0), 3)

                    confidence = math.ceil((sticker_box.conf[0]*100))/100
                    print("Confidence --->",confidence)

                    cls = int(sticker_box.cls[0])
                    print("Class name -->", sticker_classNames[cls])

                    org = [_x1, _y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(crop, sticker_classNames[cls], org, font, fontScale, color, thickness)
            # show in frame
            cv2.imshow('detected cube', crop)
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", cube_classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, cube_classNames[cls], org, font, fontScale, color, thickness)
    # cv2.imshow('frame', results.imgs[0])
    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break
