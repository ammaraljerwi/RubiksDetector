from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def get_model():
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return processor, model

def get_boxes(image, text):
    processor, model = get_model()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(outputs, 
                                                               inputs.input_ids, 
                                                               box_threshold=0.4, 
                                                               text_threshold=0.4, 
                                                               target_sizes=[im.size[::-1] for im in image])
    return results

def process_to_yolo(image, boxes):
    im = np.array(image)
    im_h, im_w = im.shape[:2]
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 /= im_w
        x2 /= im_w
        y1 /= im_h
        y2 /= im_h
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        yolo_boxes.append((0, x, y, w, h))
    return yolo_boxes

def plot_boxes(images, boxes):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))
    for i, (im, box) in enumerate(zip(images, boxes)):
        ax = axs[i]
        ax.imshow(im)
        for b in box:
            x1, y1, x2, y2 = [int(x) for x in b]
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red'))
    plt.show()
    
fp = 'simulated_cube_dataset/images/train/'
start_time = time.time()
images = [Image.open(fp + f) for i, f in enumerate(os.listdir(fp)) if f.endswith('.jpg') and i <= 5]
images = [process_image(im) for im in images]
txt = ['rubiks cube.'] * len(images)
results = get_boxes(images, txt)
end_time = time.time()
print(f'Elapsed time: {end_time - start_time}')
plot_boxes(images, [r['boxes'] for r in results])

# # im = Image.open('data/IMG_7683.jpeg')
# # im1 = process_image(im)
# # im2 = Image.open('data/IMG_7684.jpeg')
# # im2 = process_image(im2)
# # ims = [im1, im2]
# # txt = ['rubiks cube.', 'rubiks cube.']
# results = get_boxes(ims, txt)
# print(results)

# # plot_boxes(im, results[0]['boxes'])
# plot_boxes(ims, [r['boxes'] for r in results])