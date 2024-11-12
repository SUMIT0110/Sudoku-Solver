from sudoku_utils import find_puzzle, extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2
import os

def solve_sudoku(image_path, model_path='models/digit_classifier.h5', debug=False):
    # Load the model
    print("[INFO] loading digit classifier...")
    model = load_model(model_path)
    
    # Load and preprocess image
    print("[INFO] processing image...")
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    
    # Find puzzle in image
    (puzzleImage, warped) = find_puzzle(image, debug=debug)
    
    # Initialize board
    board = np.zeros((9, 9), dtype="int")
    
    # Calculate cell dimensions
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    cellLocs = []
    
    # Process each cell
    for y in range(9):
        row = []
        for x in range(9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            
            row.append((startX, startY, endX, endY))
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=debug)
            
            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred
        cellLocs.append(row)
    
    # Solve puzzle
    print("[INFO] solving Sudoku puzzle...")
    puzzle = Sudoku(3, 3, board=board.tolist())
    solution = puzzle.solve()
    
    # Create a blank puzzle grid
    solved_grid = np.ones((450, 450, 3), dtype="uint8") * 255  # 9x9 grid with each cell 50x50 pixels
    
    # Draw the solution on the blank grid
    for y in range(9):
        for x in range(9):
            # Calculate position for each digit
            startX = x * 50
            startY = y * 50
            textX = startX + 15
            textY = startY + 35
            cv2.putText(solved_grid, str(solution.board[y][x]), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Draw grid lines
    for i in range(1, 9):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(solved_grid, (0, i * 50), (450, i * 50), (0, 0, 0), thickness)
        cv2.line(solved_grid, (i * 50, 0), (i * 50, 450), (0, 0, 0), thickness)
    
    # Save the solved grid as a separate image
    output_path = os.path.join('output', 'solved_grid_' + os.path.basename(image_path))
    cv2.imwrite(output_path, solved_grid)
    
    # Display the solved grid
    cv2.imshow("Solved Sudoku Grid", solved_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return solved_grid

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input Sudoku puzzle image")
    ap.add_argument("-m", "--model", default="models/digit_classifier.h5",
                    help="path to trained digit classifier")
    ap.add_argument("-d", "--debug", type=int, default=-1,
                    help="whether or not we are visualizing each step of the pipeline")
    args = vars(ap.parse_args())
    
    solve_sudoku(args["image"], args["model"], args["debug"] > 0)
