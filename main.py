from ultralytics import YOLO
import cv2
import math
import numpy as np

from constants import CONSTANTS
from dataWriter import DataWriter
from state_manager import StateManager
from solver import convert_state_to_cube, solve_cube


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


def change_color_of_square(color, state_manager):
    global SELECTED_FACE, SELECTED_SQUARE
    if SELECTED_FACE is None or SELECTED_SQUARE is None:
        print("No face or square selected")
        return
    face_idx = SELECTED_FACE
    square_idx = convert_pair_to_idx(SELECTED_SQUARE)
    # get the state of the face
    state = state_manager.STATES[face_idx][0].cls
    # get the color of the square
    square_color = CONSTANTS.sticker_classNames[int(state[square_idx].item())]
    # change the color of the square
    if square_color == color:
        print("Square already has this color")
        return
    state_manager.STATES[face_idx][0].cls[square_idx] = CONSTANTS.color_to_idx[color]
    print(f"Changed color of square {square_idx} on face {face_idx} to {color}")
    SELECTED_FACE = None
    SELECTED_SQUARE = None


def select_square(event, x, y, flags, param):
    """mouse callback function"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")
        for color in CONSTANTS.colors.values():
            if np.array_equal(img[y, x], color):
                print(f"Color found: {color}")
                get_square(x, y)
                print(f"Selected square: {SELECTED_SQUARE}")
                print(f"Selected face: {SELECTED_FACE}")


state_handler = StateManager()
data_handler = DataWriter()
model = YOLO("models/cube_detector.pt")
sticker_labeler = YOLO("models/sticker_labeler.pt")
cap = cv2.VideoCapture(0)
box_window = cv2.namedWindow("detected cube", cv2.WINDOW_NORMAL)
main_window = cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frame", select_square)

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
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # crop frame to box
            crop = img[y1:y2, x1:x2]

            sticker_results = sticker_labeler(crop, stream=True, verbose=False)

            for sticker in sticker_results:
                sticker_boxes = sticker.boxes
                # print("labels --->", sticker_boxes.cls)
                if len(sticker_boxes.cls) == 9:
                    # save state if spacebar is pressed
                    if cv2.waitKey(1) & 0xFF == ord(" "):
                        if state_handler.save_state(sticker, crop):
                            print("saved state")
                        else:
                            print("state already saved")
                    # draw_prediction_squares(img, sticker)

                for sticker_box in sticker_boxes:
                    _x1, _y1, _x2, _y2 = sticker_box.xyxy[0]
                    _x1, _y1, _x2, _y2 = int(_x1), int(_y1), int(_x2), int(_y2)

                    cv2.rectangle(crop, (_x1, _y1), (_x2, _y2), (255, 0, 0), 3)

                    confidence = math.ceil((sticker_box.conf[0] * 100)) / 100
                    # print("Confidence --->",confidence)

                    cls = int(sticker_box.cls[0])
                    # print("Class name -->", sticker_classNames[cls])

                    org = [_x1, _y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(
                        crop,
                        CONSTANTS.sticker_classNames[cls],
                        org,
                        font,
                        fontScale,
                        color,
                        thickness,
                    )

            # show in frame
            cv2.imshow("detected cube", crop)
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
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

            cv2.putText(
                img,
                CONSTANTS.cube_classNames[cls],
                org,
                font,
                fontScale,
                color,
                thickness,
            )
    state_handler.draw_states(img)
    # cv2.imshow('frame', results.imgs[0])
    k = cv2.waitKey(1) & 0xFF
    if k == ord("w"):
        print("writing states")
        for state in state_handler.STATES.values():
            data_handler.write_data(state)
    if k == ord("1"):
        change_color_of_square(CONSTANTS.num_to_color[1], state_handler)
        print(f"Changed color of square to {CONSTANTS.num_to_color[1]}")
    if k == ord("2"):
        change_color_of_square(CONSTANTS.num_to_color[2], state_handler)
        print(f"Changed color of square to {CONSTANTS.num_to_color[2]}")
    if k == ord("3"):
        change_color_of_square(CONSTANTS.num_to_color[3], state_handler)
        print(f"Changed color of square to {CONSTANTS.num_to_color[3]}")
    if k == ord("4"):
        change_color_of_square(CONSTANTS.num_to_color[4], state_handler)
        print(f"Changed color of square to {CONSTANTS.num_to_color[4]}")
    if k == ord("5"):
        change_color_of_square(CONSTANTS.num_to_color[5], state_handler)
        print(f"Changed color of square to {CONSTANTS.num_to_color[5]}")
    if k == ord("6"):
        change_color_of_square(CONSTANTS.num_to_color[6], state_handler)
        print(f"Changed color of square to {CONSTANTS.num_to_color[6]}")

    if k == ord("c"):
        state_handler.clear_states()
        print("cleared states")
    cv2.imshow("frame", img)

    if k == ord("s"):
        if len(state_handler.STATES) != 6:
            print("Not enough states saved")
            continue
        else:
            cube = convert_state_to_cube(state_handler.STATES)
            print("Cube state: ", cube.get())
            print(cube)
            print("Solving cube...")
            solution = solve_cube(state_handler.STATES)
            print("Solution: ", solution)
            break

    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
