# sudoku_utils.py

import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

def find_puzzle(image, debug=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize lighting using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Edge detection with morphological processing
    edges = cv2.Canny(thresh, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge-detected image
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # Apply a four-point perspective transform
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    return (puzzle, warped)


def extract_digit(cell, debug=False):
    # Convert cell to grayscale (handles colored digits)
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Normalize lighting with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cell = clahe.apply(cell)

    # Apply thresholding
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None

    # Extract the largest contour (assumed to be the digit)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Calculate percentage of filled pixels
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.03 or percentFilled > 0.8:
        return None

    # Apply mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Add padding to the digit
    digit = np.pad(digit, ((4, 4), (4, 4)), 'constant', constant_values=0)
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    return digit
