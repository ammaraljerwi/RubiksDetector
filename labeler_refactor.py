import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(250 * 250 * 3, 784)
        self.fc2 = nn.Linear(784, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 6)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
model = DummyModel()

model.load_state_dict(torch.load('models/color_labeler.pth', weights_only=True, map_location=torch.device('cpu')))
LABELS = ['green', 'blue', 'orange', 'red', 'yellow', 'white']

def pred_from_contours(img: np.ndarray, model: nn.Module) -> list:
    # get bounding boxes of contours
    detected = img.copy()

    contours = get_contours(detected, return_all=False)

    yolo_labels = []
    if len(contours) == 0:
        return detected
    
    for i in range(len(contours)):
        r = cv2.boundingRect(contours[i])
        x, y, w, h = r
        cropped_im = detected[y: y+h, x: x+w]
        resized = cv2.resize(cropped_im, (250, 250))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = torch.from_numpy(resized).float()
        resized = resized.unsqueeze(0)
        pred = model(resized)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.item()
        cv2.putText(detected, LABELS[pred], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(detected, (x, y), (x + w, y + h), (255, 255, 255), 3)
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        norm_x = center_x / 250
        norm_y = center_y / 250
        norm_w = w / 250
        norm_h = h / 250
        yolo_labels.append((pred, norm_x, norm_y, norm_w, norm_h))
    
    print(yolo_labels)
    
    return detected
# data dir
data_dir = Path('data')

# COLOR RANGES

# HSV
GREEN = [[40, 50, 50], [70, 255, 255]]

# HSV
BLUE = [[90, 100, 100], [140, 255, 255]]

# HSV
ORANGE = [[5, 100, 100], [25, 255, 255]]

# HSV BUT USE RGB
RED = [[115, 100, 100], [140, 255, 255]]

# HSV
YELLOW = [[25, 110, 110], [50, 255, 255]]

# HLS
WHITE = [[0, 160, 0], [179, 255, 255]]

RANGES = [GREEN, BLUE, ORANGE, RED, YELLOW, WHITE]

def load_image(file_name, detection=False, segmentation=False) -> np.ndarray:
    if isinstance(file_name, int):
        fn = 'data/IMG_{0}.jpeg'.format(file_name) 
    else:
        fn = data_dir / file_name

    im = cv2.imread(fn)

    if detection and segmentation:
        with open('detections/rubiksdetection_{0}.pkl'.format(file_name), 'rb') as f:
            results = pickle.load(f)
        
        with open('segmentations/masks_{0}.pkl'.format(file_name), 'rb') as f:
            masks = pickle.load(f)

        return (im, results, masks)
    elif detection:
        with open('detections/rubiksdetection_{0}.pkl'.format(file_name), 'rb') as f:
            results = pickle.load(f)

        return (im, results)
    elif segmentation:
        with open('segmentations/masks_{0}.pkl'.format(file_name), 'rb') as f:
            masks = pickle.load(f)

        return (im, masks)
    
    else:
        return im


def crop_image(img: np.ndarray, results: dict, flipped=False) -> np.ndarray:
    x1, y1, x2, y2 = results[0]['boxes'][0].astype('int')
    if flipped:
        return img[x1:x2, y1:y2]
    else:
        return img[y1:y2, x1:x2]

def detect_color(img: np.ndarray, color_range: list, color_space=cv2.COLOR_BGR2HSV) -> np.ndarray:
    image = img.copy()
    original = image.copy()
    
    image = cv2.cvtColor(image, color_space)
    
    lower = np.array(color_range[0], dtype="uint8")
    upper = np.array(color_range[1], dtype="uint8")
    
    # Create a mask, erode and dilate to remove noise
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)

    detected = cv2.bitwise_and(original, original, mask=mask)

    return detected

def filter_color(img: np.ndarray, color_range: list, color_space=cv2.COLOR_BGR2HSV) -> np.ndarray:
    detected = detect_color(img, color_range, color_space)

    contours, heirarchy = get_contours(detected)

    if len(contours) == 0:
        return detected
    
    vis = np.zeros((detected.shape[0], detected.shape[1]), dtype=np.uint8)

    for i in range(len(heirarchy[0])):
        r = cv2.boundingRect(contours[i])
        if heirarchy[0][i][2] > -1:
            epsilon = 0.1 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            if len(approx) == 4:
                cv2.drawContours(vis, contours, i, (255, 255, 255), -1)

    new_vis = cv2.bitwise_and(detected, detected, mask=vis)

    return new_vis

def get_contours(img: np.ndarray, thresholds=[50,200], return_all=True, approx_poly=False) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    thresholded = cv2.threshold(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)[1]

    canny = cv2.Canny(thresholded, thresholds[0], thresholds[1])

    contours, heirarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if return_all:
        return (contours, heirarchy)
    else:
        kept_contours = []
        for i in range(len(heirarchy[0])):
            if heirarchy[0][i][2] > -1:
                kept_contours.append(contours[i])
        kept_contours = sorted(kept_contours, key=cv2.contourArea, reverse=True)
        if len(kept_contours) > 9:
            kept_contours = kept_contours[:9]
            
        return kept_contours

def visualize_contours(img: np.ndarray, contours: np.ndarray) -> None:
    vis = img.copy()

    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    return vis

    
def visualize_masks(img: np.ndarray, masks: np.ndarray) -> None:
    for i in range(len(masks)):
        mask = masks[i]['segmentation']
        mask = mask.astype(np.uint8)
        mask[mask == 1] = 255
        cv2.imshow(f'mask_{i}', mask)
        cv2.waitKey(0)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
def main():
    im, results, masks = load_image(7728, detection=True, segmentation=True)
    # print(masks)
    cropped = crop_image(im, results, flipped=False)
    
    cropped = cv2.imread('simulated_cube_dataset/images/train/0313.jpg')
    x1, y1, x2, y2 = [350, 276, 594, 491]
    cropped = cropped[y1:y2, x1:x2]
    # cropped = cv2.imread('data/sim2.png')
    cropped = cv2.resize(cropped, (500, 500))

    cv2.imshow('cropped', cropped)

    final_image = np.zeros_like(cropped)

    for i, color in enumerate(RANGES):
        if i == 3:
            # RED
            detected = filter_color(cropped, color, cv2.COLOR_RGB2HSV)
        elif i == 5:
            # WHITE
            detected = filter_color(cropped, color, cv2.COLOR_BGR2HLS)
        else:
            detected = filter_color(cropped, color)
        # cv2.imshow(f'{i}', detected)
        final_image = cv2.bitwise_or(final_image, detected)

    cv2.imshow('final', final_image)
    
    conts = get_contours(final_image, return_all=False, approx_poly=True)
    binary = np.zeros_like(final_image)
    cv2.drawContours(binary, conts, -1, (255, 255, 255), -1)
    cv2.imshow('binary', binary)
    
    # vis = visualize_contours(final_image, conts[0])
    # cv2.imshow('contours', vis)
    # cv2.imshow('contours', conts)

    pred = pred_from_contours(final_image, model)

    cv2.imshow('pred', pred)
    
    cv2.waitKey(0)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


def new_main():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.resize(frame, (500, 500))

        cv2.imshow('frame', frame)
        
        final_image = np.zeros_like(frame)
        
        for i, color in enumerate(RANGES):
            if i == 3:
                # RED
                detected = filter_color(frame, color, cv2.COLOR_RGB2HSV)
            elif i == 5:
                # WHITE
                detected = filter_color(frame, color, cv2.COLOR_BGR2HLS)
            else:
                detected = filter_color(frame, color)
            cv2.imshow(f'{i}', detected)
            final_image = cv2.bitwise_or(final_image, detected)
        cv2.imshow('final', final_image)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == '__main__':
    main()