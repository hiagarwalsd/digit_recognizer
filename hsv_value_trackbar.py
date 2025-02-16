import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Set video feed window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def nothing(x):
    pass

# Create trackbar window
cv2.namedWindow('Trackbars')
cv2.createTrackbar("L - H", "Trackbars", 35, 179, nothing)  # Lower Hue for green
cv2.createTrackbar("L - S", "Trackbars", 50, 255, nothing)  # Lower Saturation
cv2.createTrackbar("L - V", "Trackbars", 50, 255, nothing)  # Lower Value
cv2.createTrackbar("U - H", "Trackbars", 85, 179, nothing)  # Upper Hue for green
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing) # Upper Saturation
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing) # Upper Value

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # Create mask for detecting green
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask_3, frame, res))

    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

    if key == ord('s'):  # Press 's' to save the HSV values
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)
        np.save('hsv_value.npy', thearray)
        break

cap.release()
cv2.destroyAllWindows()
