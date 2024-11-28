import opencv-python-headless
# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from solve_sudoku import solve_sudoku
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🎯",
    layout="centered"
)

def load_model():
    """Load the digit recognition model"""
    model_path = 'models/digit_classifier.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure the model is properly uploaded.")
        return None
    return model_path

def main():
    st.title("Sudoku Puzzle Solver 🧩")
    st.write("Upload an image of a Sudoku puzzle and I'll solve it for you!")

    # Load model at startup
    model_path = load_model()
    if model_path is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a Sudoku puzzle image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Create a temporary directory if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Puzzle", use_container_width=True)

            
            # Save the uploaded file temporarily
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add a solve button
            if st.button("Solve Puzzle"):
                with st.spinner("Solving the puzzle..."):
                    try:
                        # Process the image
                        solved_image = solve_sudoku(temp_path, model_path)
                        
                        # Convert BGR to RGB for displaying
                        solved_image_rgb = cv2.cvtColor(solved_image, cv2.COLOR_BGR2RGB)
                        
                        # Display the result
                        st.success("Puzzle solved successfully!")
                        st.image(solved_image_rgb, caption="Solved Puzzle", use_container_width=True)                        
                        # Add download button for solved image
                        temp_solved_path = os.path.join(temp_dir, "solved_" + uploaded_file.name)
                        cv2.imwrite(temp_solved_path, solved_image)
                        with open(temp_solved_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Solved Puzzle",
                                data=file,
                                file_name="solved_" + uploaded_file.name,
                                mime="image/jpeg"
                            )
                    
                    except Exception as e:
                        st.error(f"Error solving the puzzle: {str(e)}")
                        st.info("Please make sure the image is clear and contains a valid Sudoku puzzle.")
        
        finally:
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Add information section
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload a clear image of an unsolved Sudoku puzzle
    2. Click the 'Solve Puzzle' button
    3. Wait for the solution to appear
    4. Download the solved puzzle if needed
    
    ### Tips for best results:
    - Ensure the puzzle is well-lit and clearly visible
    - The image should be properly aligned
    - Avoid glare or shadows on the puzzle
    """)

if __name__ == "__main__":
    main()
