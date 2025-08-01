# Drowsiness Detection System

A real-time drowsiness detection system that can monitor multiple people simultaneously, detect if they are sleeping, and estimate their ages. Built with PyTorch, OpenCV, and PyQt5.

## Features

- **Real-time Detection**: Works with webcam feed, video files, and images
- **Multiple Person Detection**: Can track and monitor multiple people simultaneously
- **Age Estimation**: Provides age estimates for each detected face
- **Drowsiness Detection**: Monitors eye state to detect if someone is sleeping
- **User-friendly GUI**: Easy-to-use interface with multiple input options
- **Multi-threaded Processing**: Efficient processing for smooth performance

## Requirements

```
Python 3.9
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
facenet-pytorch>=2.5.3
PyQt5>=5.15.0
scipy>=1.11.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd drowsiness-detection
```

2. Create a virtual environment:
```bash
python -m venv .venv39
.\.venv39\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Files

The system requires two pre-trained model files:
- `best_model.pth`: Age estimation model
- `eye_detector.pth`: Eye state detection model

Place these files in the root directory of the project.

## Usage

1. Run the application:
```bash
python drowsiness_detection_gui.py
```

2. Use the GUI buttons to:
   - **Open Webcam**: Start real-time detection using your webcam. Doesn't working fine as of now, require more fine tuning
   - **Open Video**: Select and process a video file
   - **Open Image**: Select and process an image file
   - **Stop**: Stop the current detection session

## Features in Detail

### Face Detection
- Uses MTCNN for robust face detection
- Tracks faces across frames
- Maintains consistent IDs for each person

### Age Estimation
- Uses ResNet50 model for accurate age prediction
- Displays age above each detected face
- Real-time updates for video streams

### Drowsiness Detection
- Monitors eye state using ResNet18 model
- Marks sleeping people with red bounding boxes
- Shows "Sleeping" label for detected drowsy states

### Performance Optimizations
- Multi-threaded processing for smooth performance
- Frame queue management to prevent lag
- Efficient face tracking and caching
- Optimized for real-time processing

## Project Structure

- `drowsiness_detection_gui.py`: Main application file with GUI implementation
- `requirements.txt`: List of Python dependencies
- `best_model.pth`: Pre-trained age estimation model
- `eye_detector.pth`: Pre-trained eye state detection model

## Technical Implementation

### Models Used
1. **MTCNN**: Face detection and facial landmarks
2. **ResNet50**: Age estimation
3. **ResNet18**: Eye state classification

### Key Components
- **Frame Processor**: Handles image processing in a separate thread
- **Face Cache**: Maintains face tracking information
- **Queue System**: Manages frame processing pipeline

## Limitations

- Performance depends on hardware capabilities
- Best results with good lighting conditions
- Requires clear view of faces and eyes
- Processing speed may vary with number of faces

## Future Improvements

- Add support for GPU acceleration
- Implement face recognition
- Add alert sound for drowsiness detection
- Add statistics and logging features
- Improve low-light performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
