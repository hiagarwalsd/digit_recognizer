import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import DigitCNN

# Load the trained model
model = DigitCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# OpenCV setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
kernel = np.ones((5, 5), np.uint8)
x1, y1 = 0, 0
noise_thresh = 800

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Load color range (assumes hsv_value.npy exists)
    hsv_value = np.load('hsv_value.npy')
    lower_range, upper_range = hsv_value[0], hsv_value[1]
    
    # Masking
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)
        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 8)
        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0
    
    # Add canvas to frame
    frame = cv2.add(frame, canvas)
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Live Writing', cv2.resize(stacked, None, fx=0.6, fy=0.6))
    
    # Key press actions
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    elif key == ord('p'):
        # Crop and preprocess digit
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        digit = thresh[y:y+h, x:x+w]
        
        # Center and resize the digit
        h, w = digit.shape
        pad_size = max(h, w)
        padded_digit = np.zeros((pad_size, pad_size), dtype=np.uint8)
        x_offset = (pad_size - w) // 2
        y_offset = (pad_size - h) // 2
        padded_digit[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        digit = cv2.resize(padded_digit, (28, 28))
        
        # Show the preprocessed input image
        cv2.imshow('Model Input', digit)
        cv2.waitKey(500)  # Display for 500ms
        
        # Normalize input
        digit = digit.astype(np.float32) / 255.0
        digit = torch.tensor(digit).unsqueeze(0).unsqueeze(0)
        
        # Predict digit
        with torch.no_grad():
            output = model(digit)
            prediction = torch.argmax(output, dim=1).item()
        
        print(f"Predicted Digit: {prediction}")
        cv2.putText(frame, f"Predicted: {prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cap.release()
cv2.destroyAllWindows()
