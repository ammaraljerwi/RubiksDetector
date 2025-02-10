import cv2
import sys
import numpy as np
import pickle

with open('detections/rubiksdetection_7683.pkl', 'rb') as f:
    results = pickle.load(f)

def nothing(x):
    pass

# Load in image
image = cv2.imread('data/IMG_7683.jpeg')

x1, y1, x2, y2 = results[0]['boxes'][0].astype('int')

image = image[x1:x2, y1:y2]
# image = image[y1:y2, x1:x2]

# image = cv2.imread('data/simss.png')
b = np.uint8([[[255,0,0]]])
g = np.uint8([[[0,255,0]]])
r = np.uint8([[[0,0,255]]])
w = np.uint8([[[255,255,255]]])

hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
hsv_g = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
hsv_r = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
hsv_w = cv2.cvtColor(w, cv2.COLOR_BGR2HSV)

print("HSV for BLUE: ", hsv_b)
print("HSV for GREEN: ", hsv_g)
print("HSV for RED: ", hsv_r)
print("HSV for WHITE: ", hsv_w)

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 1

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()