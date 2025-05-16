from ultralytics import YOLO
import cv2
import math
import torch
import numpy as np
from sklearn.cluster import KMeans

import magiccube

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

def draw_prediction_squares(img, results, sorted_preds=None, start_x=0, start_y=0):
   
   # get top left corner of frame
    x, y = start_x, start_y

    if sorted_preds is None:
      ordered_classes, _ = sort_prediction(results)
      if ordered_classes is None:
         return
    else:
      ordered_classes = sorted_preds
    # create 3x3 grid of squares starting at top left corner
    for i in range(3):
        for j in range(3):
            # get color of square
            color = int(ordered_classes[i*3+j].item())
            # draw square
            cv2.rectangle(img, (x+j*100, y+i*100), (x+j*100+100, y+i*100+100), colors[sticker_classNames[color]], -1)
            # draw border
            cv2.rectangle(img, (x+j*100, y+i*100), (x+j*100+100, y+i*100+100), (0, 0, 0), 2)

SAVED_STATES = []
SAVED_SORTED_STATES = []

def save_state(state):
  """Save current face to list"""
  sorted_classes, _ = sort_prediction(state)
  if sorted_classes is None:
     return
  for i in range(len(SAVED_STATES)):
     if torch.equal(sorted_classes, SAVED_SORTED_STATES[i]):
        return

  SAVED_STATES.append(state)
  SAVED_SORTED_STATES.append(sorted_classes)


def draw_states(img):
   """draw faces on top of screen, 6 3x3 faces horizontally"""
   x, y = 0, 0
   for i, state in enumerate(SAVED_STATES):
      draw_prediction_squares(img, state, SAVED_SORTED_STATES[i], x, y)
      x += 400
      if x > img.shape[1]:
         x = 0
         y += 400
         if y > img.shape[0]:
            break
         

def convert_state_to_cube(state):
  state_str = ""
  for i in range(len(state)):
     for j in range(len(state[i])):
        state_str += idx_to_color[int(state[i][j].item())]
  # cube = magiccube.Cube(3, state_str)
  return state_str

def solve_cube(state):
  """solve cube"""
  cube = convert_state_to_cube(state).lower()
  # solver = magiccube.BasicSolver(cube)
  # solution = solver.solve()
  solution = utils.solve(cube, method='Kociemba')
  return solution
  
def clear_states():
   """clear saved states"""
   global SAVED_STATES
   SAVED_STATES = []

def faces_per_row(img):
    """return number of faces that can fit in image"""
    return img.shape[1] // 400 + 1

SELECTED_FACE = None
SELECTED_SQUARE = None

def convert_pair_to_idx(pair):
    """Convert (row, col) pair to index"""
    return pair[0] * 3 + pair[1]

def convert_idx_to_pair(idx):
    """Convert index to (row, col) pair"""
    return (idx // 3, idx % 3)

def get_square(x, y):
    global SELECTED_FACE, SELECTED_SQUARE
    face_idx = 0
    # check which face is selected
    while x > 400:
        x -= 400
        face_idx += 1
    while y > 400:
        y -= 400
        face_idx += faces_per_row(img)
    SELECTED_FACE = face_idx
    print(f"Face selected: {face_idx}")
    # each square is 100x100, (0-100, 0-100) is top left, (100-200, 0-100) is top right, etc.
    x_square = math.floor(x // 100)
    y_square = math.floor(y // 100)
    SELECTED_SQUARE = (y_square, x_square)
    print(f"Square selected: ({y_square}, {x_square})")

def change_color_of_square(color):
  global SELECTED_FACE, SELECTED_SQUARE
  if SELECTED_FACE is None or SELECTED_SQUARE is None:
    print("No face or square selected")
    return
  face_idx = SELECTED_FACE
  square_idx = convert_pair_to_idx(SELECTED_SQUARE)
  # get the state of the face
  state = SAVED_SORTED_STATES[face_idx]
  # get the color of the square
  square_color = sticker_classNames[int(state[square_idx].item())]
  # change the color of the square
  if square_color == color:
    print("Square already has this color")
    return
  SAVED_SORTED_STATES[face_idx][square_idx] = color_to_idx[color]
  print(f"Changed color of square {square_idx} on face {face_idx} to {color}")
  SELECTED_FACE = None
  SELECTED_SQUARE = None
  

def select_square(event, x, y, flags, param):
  """mouse callback function"""
  if event == cv2.EVENT_LBUTTONDOWN:
      print(f"Mouse clicked at ({x}, {y})")
      for color in colors.values():
          if np.array_equal(img[y, x], color):
              print(f"Color found: {color}")
              get_square(x, y)
              print(f"Selected square: {SELECTED_SQUARE}")
              print(f"Selected face: {SELECTED_FACE}")

         


model = YOLO('models/cube_detector.pt')
sticker_labeler = YOLO('models/sticker_labeler.pt')
cube_classNames = ["Rubik's Cube"]
sticker_classNames = ['green', 'blue', 'orange', 'red', 'yellow', 'white']
idx_to_color = {0: 'G', 1: 'B', 2: 'O', 3: 'R', 4: 'Y', 5: 'W'}
colors = {
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'orange': (0, 165, 255),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255)
}
rgb_to_color = {v: k for k, v in colors.items()}
color_to_idx = {k: i for i, k in enumerate(colors.keys())}
num_to_color = {(i+1): k for i, k in enumerate(colors.keys())}
cap = cv2.VideoCapture(0)
box_window = cv2.namedWindow('detected cube', cv2.WINDOW_NORMAL)
main_window = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', select_square)

while True:
    ret, img = cap.read()
    # results = model(img)

    results = model(img, stream=True, verbose=False)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # crop frame to box
            crop = img[y1:y2, x1:x2]

            sticker_results = sticker_labeler(crop, stream=True, verbose=False)
            
            for sticker in sticker_results:
                sticker_boxes = sticker.boxes
                # print("labels --->", sticker_boxes.cls)
                if len(sticker_boxes.cls) == 9:
                    # save state if spacebar is pressed
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                      save_state(sticker)
                      print("saved state")
                    # draw_prediction_squares(img, sticker)
                

                for sticker_box in sticker_boxes:
                    _x1, _y1, _x2, _y2 = sticker_box.xyxy[0]
                    _x1, _y1, _x2, _y2 = int(_x1), int(_y1), int(_x2), int(_y2)
                    
                    cv2.rectangle(crop, (_x1, _y1), (_x2, _y2), (255, 0, 0), 3)

                    confidence = math.ceil((sticker_box.conf[0]*100))/100
                    # print("Confidence --->",confidence)

                    cls = int(sticker_box.cls[0])
                    # print("Class name -->", sticker_classNames[cls])

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
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            # print("Class name -->", cube_classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, cube_classNames[cls], org, font, fontScale, color, thickness)
    # cv2.imshow('frame', results.imgs[0])
    draw_states(img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('1'):
        change_color_of_square(num_to_color[1])
        print(f"Changed color of square to {num_to_color[1]}")
    if k == ord('2'):
        change_color_of_square(num_to_color[2])
        print(f"Changed color of square to {num_to_color[2]}")
    if k == ord('3'):
        change_color_of_square(num_to_color[3])
        print(f"Changed color of square to {num_to_color[3]}")
    if k == ord('4'):
        change_color_of_square(num_to_color[4])
        print(f"Changed color of square to {num_to_color[4]}")
    if k == ord('5'):
        change_color_of_square(num_to_color[5])
        print(f"Changed color of square to {num_to_color[5]}")
    if k == ord('6'):
        change_color_of_square(num_to_color[6])
        print(f"Changed color of square to {num_to_color[6]}")
      
    if k == ord('c'):
        clear_states()
        print("cleared states")
    cv2.imshow('frame', img)

    if k == ord('s'):
      if len(SAVED_SORTED_STATES) != 6:
        print("Not enough states saved")
        continue
      else:
        cube = convert_state_to_cube(SAVED_SORTED_STATES)
        print("Cube state: ", cube.get())
        print(cube)
        print("Solving cube...")
        solution = solve_cube(SAVED_SORTED_STATES)
        print("Solution: ", solution)
        break


    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
