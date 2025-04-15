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
image = cv2.resize(image, (500,500))

b = np.uint8([[[255,0,0]]])
w = np.uint8([[[255,255,255]]])

hls_b = cv2.cvtColor(b, cv2.COLOR_BGR2HLS)
hls_w = cv2.cvtColor(w, cv2.COLOR_BGR2HLS)

print("HLS for BLUE: ", hls_b)
print("HLS for WHITE: ", hls_w)

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('LMin','image',0,255,nothing)
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('LMax','image',0,255,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('LMax', 'image', 255)
cv2.setTrackbarPos('SMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = lMin = sMin = hMax = lMax = sMax = 0
phMin = plMin = psMin = phMax = plMax = psMax = 0

output = image
wait_time = 1

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    lMin = cv2.getTrackbarPos('LMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    lMax = cv2.getTrackbarPos('LMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, lMin, sMin])
    upper = np.array([hMax, lMax, sMax])

    # Create HSV Image and threshold into a range.
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (plMin != lMin) | (psMin != sMin) | (phMax != hMax) | (plMax != lMax) | (psMax != sMax) ):
        print("(hMin = %d , lMin = %d, sMin = %d), (hMax = %d , lMax = %d, sMax = %d)" % (hMin , lMin , sMin, hMax, lMax , sMax))
        phMin = hMin
        plMin = lMin
        psMin = sMin
        phMax = hMax
        plMax = lMax
        psMax = sMax

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()