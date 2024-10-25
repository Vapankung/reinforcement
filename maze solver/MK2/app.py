# app.py

from flask import Flask, render_template, jsonify, request
import numpy as np
from maze_env import MazeEnv
from dql_agent import DQNAgent
from werkzeug.utils import secure_filename
import os
import cv2
import imutils
from scipy.signal import find_peaks

app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_maze_image(image_path):
    """
    Process the uploaded maze image to extract the maze grid.
    This function attempts to detect the maze size automatically and convert the image to a binary grid.
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unsupported format.")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better segmentation
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        if not contours:
            raise ValueError("No contours found in image.")

        # Assume the largest contour is the maze
        c = max(contours, key=cv2.contourArea)

        # Get the bounding box of the maze
        x, y, w, h = cv2.boundingRect(c)
        maze_roi = thresh[y:y+h, x:x+w]

        # Resize the image to a standard size (e.g., 100x100) for processing
        resized = cv2.resize(maze_roi, (100, 100), interpolation=cv2.INTER_AREA)

        # Detect vertical and horizontal lines to estimate cell size
        horizontal_sum = np.sum(resized, axis=1)
        vertical_sum = np.sum(resized, axis=0)

        # Use peaks in the sums to detect lines
        h_peaks, _ = find_peaks(horizontal_sum, distance=5)
        v_peaks, _ = find_peaks(vertical_sum, distance=5)

        # Estimate the number of cells
        num_cells = len(h_peaks)
        if num_cells < 2:
            num_cells = 10  # Default to 10x10 if detection fails

        # Resize the maze to num_cells x num_cells
        maze_resized = cv2.resize(maze_roi, (num_cells, num_cells), interpolation=cv2.INTER_NEAREST)

        # Threshold again to get binary values
        _, maze_binary = cv2.threshold(maze_resized, 128, 255, cv2.THRESH_BINARY)

        # Convert to a maze grid (0: open, 1: wall)
        maze = (maze_binary == 255).astype(int)

        return maze
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/solve_maze', methods=['POST'])
def solve_maze():
    """
    Solve the provided maze using the DQN agent.
    Expects a JSON payload with the maze grid.
    """
    try:
        data = request.get_json()
        maze_list = data.get('maze')

        if not maze_list:
            return jsonify({'error': 'No maze data provided.'}), 400

        maze = np.array(maze_list)
        env = MazeEnv(maze)
        action_size = env.action_space.n
        agent = DQNAgent(action_size)

        # Determine maze size
        input_shape = maze.shape

        # Construct the model path based on maze size
        model_path = f'dql_maze_solver_{input_shape[0]}x{input_shape[1]}.pth'

        # Check if the model file exists
        if not os.path.exists(model_path):
            return jsonify({'error': f'No trained model available for {input_shape[0]}x{input_shape[1]} mazes.'}), 400

        # Load the trained model
        agent.load(model_path, input_shape)
        agent.epsilon = 0.0  # Disable exploration

        # Solve the maze
        state = env.reset()
        done = False
        path = [{'x': env.agent_pos[1], 'y': env.agent_pos[0]}]
        max_steps = maze.size * 4  # Prevent infinite loops
        steps = 0

        while not done and steps < max_steps:
            state_normalized = state / 2.0  # Normalize the state
            action = agent.act(state_normalized)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            path.append({'x': env.agent_pos[1], 'y': env.agent_pos[0]})
            steps += 1

        if not done:
            return jsonify({'error': 'Agent could not find a path to the goal.'}), 400

        return jsonify({'path': path})

    except Exception as e:
        print(f"Error during maze solving: {e}")
        return jsonify({'error': 'An error occurred while solving the maze.'}), 500

@app.route('/upload_maze', methods=['POST'])
def upload_maze():
    """
    Handle the uploaded maze image, process it, and return the maze grid.
    """
    try:
        if 'maze_image' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400

        file = request.files['maze_image']

        if file.filename == '':
            return jsonify({'error': 'No selected file.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload PNG, JPG, JPEG, or BMP files.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image to extract the maze
        maze = process_maze_image(filepath)

        if maze is None:
            return jsonify({'error': 'Failed to process image. Please ensure the image is clear and properly formatted.'}), 500

        # Remove the uploaded file after processing
        os.remove(filepath)

        return jsonify({'maze': maze.tolist()})

    except Exception as e:
        print(f"Error during maze upload: {e}")
        return jsonify({'error': 'An error occurred during file upload.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
