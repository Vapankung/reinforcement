// static/script.js

const canvas = document.getElementById('mazeCanvas');
const ctx = canvas.getContext('2d');
const solveButton = document.getElementById('solveButton');
const clearButton = document.getElementById('clearButton');
const randomButton = document.getElementById('randomButton');
const mazeSizeSelect = document.getElementById('mazeSize');
const difficultySelect = document.getElementById('difficulty');
const mazeImageInput = document.getElementById('mazeImageInput');
const messagePara = document.getElementById('message');

let mazeSize = parseInt(mazeSizeSelect.value);
let difficulty = parseFloat(difficultySelect.value);
let mazeData = createEmptyMaze(mazeSize);
let cellSize = canvas.width / mazeSize;
let isDrawing = false;

// Initialize maze
drawMaze();

// Event listeners for drawing on canvas
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    toggleWall(e);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        toggleWall(e);
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

// Event listeners for buttons
clearButton.addEventListener('click', () => {
    clearMaze();
    hideMessage();
});

randomButton.addEventListener('click', () => {
    generateRandomMaze();
    hideMessage();
});

solveButton.addEventListener('click', () => {
    hideMessage();
    drawMaze();
    solveMaze();
});

// Event listener for maze size change
mazeSizeSelect.addEventListener('change', () => {
    mazeSize = parseInt(mazeSizeSelect.value);
    cellSize = canvas.width / mazeSize;
    mazeData = createEmptyMaze(mazeSize);
    drawMaze();
    hideMessage();
});

// Event listener for difficulty change
difficultySelect.addEventListener('change', () => {
    difficulty = parseFloat(difficultySelect.value);
    generateRandomMaze();
    hideMessage();
});

// Event listener for image upload
mazeImageInput.addEventListener('change', handleImageUpload);

// Functions
function createEmptyMaze(size) {
    return Array.from({ length: size }, () => Array(size).fill(0));
}

function drawMaze() {
    for (let y = 0; y < mazeData.length; y++) {
        for (let x = 0; x < mazeData[y].length; x++) {
            if (mazeData[y][x] === 1) {
                ctx.fillStyle = '#000';
            } else {
                ctx.fillStyle = '#fff';
            }
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

function drawAgent(position) {
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    ctx.arc(
        position.x * cellSize + cellSize / 2,
        position.y * cellSize + cellSize / 2,
        cellSize / 3,
        0,
        2 * Math.PI
    );
    ctx.fill();
}

function toggleWall(event) {
    const pos = getMousePosition(event);
    if (pos.x >= 0 && pos.x < mazeSize && pos.y >= 0 && pos.y < mazeSize) {
        // Prevent toggling start and goal positions
        if ((pos.x === 0 && pos.y === 0) || (pos.x === mazeSize - 1 && pos.y === mazeSize - 1)) {
            return;
        }
        mazeData[pos.y][pos.x] = mazeData[pos.y][pos.x] === 1 ? 0 : 1;
        drawMaze();
    }
}

function clearMaze() {
    mazeData = createEmptyMaze(mazeSize);
    drawMaze();
}

function generateRandomMaze() {
    mazeData = createEmptyMaze(mazeSize);
    // Simple random walls based on difficulty
    for (let y = 0; y < mazeSize; y++) {
        for (let x = 0; x < mazeSize; x++) {
            if ((x === 0 && y === 0) || (x === mazeSize - 1 && y === mazeSize - 1)) {
                continue; // Keep start and goal positions free
            }
            if (Math.random() < difficulty) {
                mazeData[y][x] = 1;
            }
        }
    }
    drawMaze();
}

async function solveMaze() {
    try {
        const response = await fetch('/solve_maze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ maze: mazeData })
        });

        if (!response.ok) {
            const errorData = await response.json();
            showMessage('Error: ' + errorData.error);
            return;
        }

        const data = await response.json();
        const path = data.path;
        animatePath(path);
    } catch (error) {
        showMessage('Error: Unable to solve the maze.');
        console.error('Error:', error);
    }
}

function animatePath(path) {
    let index = 0;

    function animate() {
        if (index < path.length) {
            drawMaze();
            drawPath(path.slice(0, index + 1)); // Draw the path
            drawAgent(path[index]);
            index++;
            setTimeout(animate, 200);
        }
    }

    animate();
}

function drawPath(path) {
    ctx.fillStyle = 'lightblue';
    for (const pos of path) {
        ctx.fillRect(pos.x * cellSize, pos.y * cellSize, cellSize, cellSize);
    }
}

function getMousePosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: Math.floor((event.clientX - rect.left) / cellSize),
        y: Math.floor((event.clientY - rect.top) / cellSize)
    };
}

async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // Create FormData to send the image to the server
        const formData = new FormData();
        formData.append('maze_image', file);

        // Send the image to the server for processing
        try {
            const response = await fetch('/upload_maze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                mazeData = data.maze;
                mazeSize = mazeData.length;
                cellSize = canvas.width / mazeSize;
                mazeSizeSelect.value = mazeSize;
                mazeSizeSelect.dispatchEvent(new Event('change')); // Trigger change event
                drawMaze();
                showMessage('Maze uploaded and processed successfully.');
            } else {
                showMessage('Error: ' + data.error);
            }
        } catch (error) {
            showMessage('Error: Unable to upload and process the image.');
            console.error('Error:', error);
        }
    }
}

function showMessage(msg) {
    messagePara.textContent = msg;
    messagePara.classList.remove('hidden');
}

function hideMessage() {
    messagePara.textContent = '';
    messagePara.classList.add('hidden');
}
