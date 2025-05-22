import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
from ultralytics.engine import results

from constants import CONSTANTS


def sort_result(results) -> results.Boxes:
    """
    From yolo results object class, return sorted prediction array and associated indices
    """
    clusters = KMeans(n_clusters=3).fit_predict(
        results.boxes.xywhn[:, 1].reshape(-1, 1)
    )

    order = np.argsort(clusters)
    clustered = torch.cat(
        (results.boxes.xywhn[order], torch.Tensor(order).unsqueeze(1)), dim=1
    )
    min_val, max_val = torch.min(clustered[:, 1]), torch.max(clustered[:, 1])

    groups = [clustered[:3], clustered[3:6], clustered[6:]]

    rows = [torch.Tensor() for _ in range(3)]
    for group in groups:
        if abs(group[0, 1] - min_val) <= 0.1:
            rows[0] = group
        elif abs(group[0, 1] - max_val) <= 0.1:
            rows[2] = group
        else:
            rows[1] = group

    for i, row in enumerate(rows):
        ids = torch.argsort(row[:, 0], descending=False)
        rows[i] = row[ids]

    sorted = torch.cat(tuple(rows), dim=0)
    idxs = sorted[:, 4].to(dtype=torch.int)

    return results.boxes[idxs]


def draw_prediction_squares(img, results, start_x=0, start_y=0):
    # get top left corner of frame
    x, y = start_x, start_y

    ordered_classes = results.cls
    # create 3x3 grid of squares starting at top left corner
    for i in range(3):
        for j in range(3):
            # get color of square
            color = int(ordered_classes[i * 3 + j].item())
            # draw square
            cv2.rectangle(
                img,
                (x + j * 100, y + i * 100),
                (x + j * 100 + 100, y + i * 100 + 100),
                CONSTANTS.colors[CONSTANTS.sticker_classNames[color]],
                -1,
            )
            # draw border
            cv2.rectangle(
                img,
                (x + j * 100, y + i * 100),
                (x + j * 100 + 100, y + i * 100 + 100),
                (0, 0, 0),
                2,
            )
