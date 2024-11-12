# sudoku_utils.py
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

class SudokuNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential([
            # First CONV => RELU => CONV => RELU => POOL layer
            Conv2D(32, (5, 5), padding="same", input_shape=(height, width, depth)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (5, 5), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Second CONV => RELU => CONV => RELU => POOL layer
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            
            # First set of FC => RELU layers
            Flatten(),
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.5),
            
            # Second set of FC => RELU layers
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.5),
            
            # Softmax classifier
            Dense(classes),
            Activation("softmax")
        ])
        return model


def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=2)

    thresh = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break
    
    if puzzleCnt is None:
        raise Exception("Could not find Sudoku puzzle outline.")
    
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    
    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    
    return (puzzle, warped)

def extract_digit(cell, debug=False):
    blurred = cv2.GaussianBlur(cell, (3, 3), 1)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return None
    
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    if percentFilled < 0.03 or percentFilled > 0.8:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    digit = np.pad(digit, ((4,4), (4,4)), 'constant', constant_values=0)
    
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    return digit