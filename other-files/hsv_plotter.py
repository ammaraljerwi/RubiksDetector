import cv2
import sys
import numpy as np
import pickle


def nothing(x):
    pass

image = np.zeros((500, 500, 3), np.uint8)

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
cv2.createTrackbar('H','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('S','image',0,255,nothing)
cv2.createTrackbar('V','image',0,255,nothing)



# Initialize to check if HSV min/max value changes
h = s = v  = 0
ph = ps = pv  = 0

output = image
wait_time = 1

while(1):

    # get current positions of all trackbars
    h = cv2.getTrackbarPos('H','image')
    s = cv2.getTrackbarPos('S','image')
    v = cv2.getTrackbarPos('V','image')


    # Create HSV Image and threshold into a range.
    hsv = [h, s, v]
    image[:,: ] = hsv
    output = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # Print if there is a change in HSV value
    if( (ph != h) | (ps != s) | (pv != v) ):
        print("(h = %d , s = %d, v = %d)" % (h, s, v))
        ph = h
        ps = s
        pv = v


    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()