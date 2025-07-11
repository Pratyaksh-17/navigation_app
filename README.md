# Navigation App

A real-time smart navigation system designed to assist visually impaired users by providing obstacle detection and directional guidance using computer vision and AI.

## Overview

This project implements a navigation aid leveraging object detection with YOLOv4-Tiny and real-time video streaming to identify obstacles and guide users safely. It uses Flask as the backend server to process video frames, detect obstacles, and send navigation instructions through a user-friendly interface.

## Features

- **Real-time video streaming** from webcam or video source.
- **Obstacle detection** using YOLOv4-Tiny deep learning model.
- **Directional guidance** such as "move left," "move right," or "stop" based on detected obstacles.
- **Audio and visual feedback** to enhance navigation assistance.
- Lightweight and optimized for efficient performance.
- Easy to deploy on local machines or edge devices.

## Technologies Used

- Python 3.x
- Flask (for backend API and streaming)
- OpenCV (for video processing)
- YOLOv4-Tiny (object detection model)
- TensorFlow Lite (optional, for optimized detection)
- WebSocket (for real-time communication)
- HTML/CSS/JavaScript (frontend UI)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Pratyaksh-17/navigation_app.git
   cd navigation_app
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv4-Tiny weights and configuration files** and place them in the appropriate folder (refer to the project structure or instructions in the code).

## Usage

1. Run the Flask app:

   ```bash
   python navigation_app.py
   ```

2. Open your browser and navigate to:

   ```
   http://localhost:5000
   ```

3. The app will start streaming video, detect obstacles in real-time, and provide navigation guidance both visually and via audio.

## Project Structure

```
navigation_app/
├── static/                 # Static assets like CSS, JS, audio files
├── templates/              # HTML templates
├── models/                 # YOLO weights and config files
├── navigation_app.py       # Main Flask application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Contribution

Contributions, suggestions, and bug reports are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Created by Pratyaksh Gupta 

- GitHub: [@Pratyaksh-17](https://github.com/Pratyaksh-17)

---

*This project is aimed at making navigation safer and easier for visually impaired users through AI-powered assistance.*
