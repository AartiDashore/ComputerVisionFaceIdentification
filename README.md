# Computer Vision Face Identification

This project is a real-time face analysis application built using **Python**, **OpenCV**, and **DeepFace**. The system captures video from a webcam, detects faces, and performs AI-based analysis to estimate **age, gender, emotion, race**, and a basic **face shape approximation**.

## Features

* Real-time webcam-based face detection and analysis
* AI-powered facial attribute estimation using DeepFace
* Detection of:

  * Age
  * Gender
  * Dominant emotion
  * Dominant race
  * Face shape (simple geometric approximation)
* Performance optimization by analyzing every N frames
* Bounding boxes and labeled attributes displayed on detected faces
* Graceful exit using keyboard input or window close event

## Technologies Used

* Python 3
* OpenCV (`cv2`)
* DeepFace
* TensorFlow / PyTorch (used internally by DeepFace)

## Project Structure

```
.
├── face_analysis.py
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install opencv-python deepface
```

> DeepFace may install additional dependencies automatically, such as TensorFlow, PyTorch, and face detection backends.

## Usage

Run the application with:

```bash
python face_analysis.py
```

## Configuration

The following variable controls how frequently facial analysis is performed:

```python
ANALYZE_EVERY_N_FRAMES = 30
```

* Lower values increase accuracy but reduce FPS
* Higher values improve performance with slightly delayed updates

## How It Works

1. Captures video frames from the webcam
2. Converts frames from BGR to RGB format
3. Runs DeepFace analysis at a configurable frame interval
4. Caches analysis results to improve performance
5. Draws bounding boxes and facial attributes on the video stream
6. Displays the processed video in real time

## Notes and Limitations

* Adequate lighting significantly improves accuracy
* Face shape detection is a basic geometric approximation
* Predictions are probabilistic and may not always be accurate
* Performance depends on system hardware

## Disclaimer

This project is intended for educational and experimental purposes only.
Facial analysis results should not be used for identification, profiling, or decision-making in sensitive contexts.

## License

This project is licensed under MIT License and provided for educational use.
