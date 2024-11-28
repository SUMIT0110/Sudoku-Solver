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
    
    # Load and preprocess the input image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)

    # Find puzzle in image
    (puzzleImage, warped) = find_puzzle(image)
    
    # Initialize the Sudoku board
    board = np.zeros((9, 9), dtype="int")

    # Calculate cell dimensions
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # Extract digits and fill the board
    for y in range(9):
        for x in range(9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell)

            if digit is not None:
                roi = digit.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=0)
                pred = model.predict(roi, verbose=0).argmax(axis=1)[0]
                board[y, x] = pred

    # Solve the puzzle and overlay the solution
    puzzle = Sudoku(3, 3, board=board.tolist())
    solution = puzzle.solve()

    for y in range(9):
        for x in range(9):
            if board[y, x] == 0:
                textX = int((x + 0.5) * stepX)
                textY = int((y + 0.7) * stepY)
                cv2.putText(puzzleImage, str(solution.board[y][x]),
                            (textX, textY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

    return puzzleImage
