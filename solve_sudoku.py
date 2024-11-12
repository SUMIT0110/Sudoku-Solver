# solve_sudoku.py

from sudoku_utils import find_puzzle, extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import imutils
import cv2

def solve_sudoku(image_path, model_path='models/digit_classifier.h5'):
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    
    # Find puzzle in image
    (puzzleImage, warped) = find_puzzle(image)
    
    # Initialize board
    board = np.zeros((9, 9), dtype="int")
    
    # Calculate cell dimensions
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    
    # Process each cell
    for y in range(9):
        for x in range(9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell)
            
            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                pred = model.predict(roi, verbose=0).argmax(axis=1)[0]
                board[y, x] = pred
    
    # Print OCR'd board
    puzzle = Sudoku(3, 3, board=board.tolist())
    
    # Solve puzzle
    solution = puzzle.solve()
    
    # Draw solution on image
    for y in range(9):
        for x in range(9):
            if board[y, x] == 0:  # Only fill in empty cells
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                
                textX = int((endX - startX) * 0.33) + startX
                textY = int((endY - startY) * 0.7) + startY
                
                cv2.putText(puzzleImage, str(solution.board[y][x]),
                    (textX, textY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
    
    return puzzleImage
