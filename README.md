# Face Recognition Demo

A real-time face recognition application built with Python, InsightFace, and Streamlit. This project provides a web-based interface for enrolling users and recognizing faces through a webcam stream.

## Features

- **Real-time Face Recognition**: Detect and recognize faces in real-time using webcam
- **User Enrollment**: Easy enrollment system with multi-image support for better accuracy
- **Face Detection**: Powered by InsightFace's state-of-the-art face detection models
- **Web Interface**: User-friendly Streamlit-based web interface
- **Similarity Scoring**: Display confidence percentages for face recognition
- **Multi-face Detection**: Support for detecting and recognizing multiple faces simultaneously
- **Persistent Storage**: Save and load face embeddings from disk
- **Optimized Performance**: Async processing and throttled inference for smooth video streaming

## Technology Stack

- **InsightFace**: Face detection and recognition
- **Streamlit**: Web interface
- **OpenCV**: Image processing
- **NumPy**: Numerical operations
- **WebRTC**: Real-time video streaming (streamlit-webrtc)
- **ONNX Runtime**: Model inference

## Project Structure

```
face-emotion/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── assets/
│   └── icons/                  # Emotion icons (if used)
├── db/
│   └── embeddings/             # Stored face embeddings (.npy files)
├── models/                     # InsightFace models (auto-downloaded)
└── src/
    ├── __init__.py
    ├── emotion_ui.py           # UI utilities for emotion display
    ├── face_engine.py          # Core face recognition engine
    ├── model/
    │   ├── __init__.py
    │   └── recognition_result.py  # Recognition result data model
    └── utils/
        ├── __init__.py
        └── util.py             # Utility functions (l2norm, cosine similarity)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Webcam for real-time recognition

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd face-emotion
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   uv venv .uv_venv/ --python 3.12 
   source uv_venv/bin/activate  # On Windows: uv_venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

   The key dependencies include:
   - `insightface==0.7.3`: Face recognition
   - `streamlit`: Web interface
   - `opencv-python`: Image processing
   - `onnxruntime`: Model inference
   - `streamlit-webrtc`: WebRTC support

## Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Enrolling Users

1. Navigate to the sidebar on the left
2. Enter the user's name in the "Tên user" field
3. Upload one or more images (JPG/PNG) of the person
   - Multiple images improve recognition accuracy
   - Ensure faces are clearly visible
4. Click "Enroll" to save the user
5. The system will extract face embeddings and save them to `db/embeddings/`

### Recognizing Faces

1. Click "Start" on the camera widget to begin the webcam stream
2. The system will automatically detect and recognize faces
3. Recognized faces will display:
   - Green bounding box around the face
   - Name of the recognized person
   - Confidence percentage

### Managing the Database

- **Reload Database**: Click "Reload DB" in the sidebar to refresh the list of known users
- **View Known Users**: The sidebar displays all enrolled users
- **Manual Management**: Face embeddings are stored as `.npy` files in `db/embeddings/`
  - Delete files to remove users
  - The filename (without extension) is the user's name

## Configuration

### Face Recognition Engine

In [app.py](app.py), you can configure the `FaceEngine`:

```python
engine = FaceEngine(
    model_name="buffalo_s",              # Model: buffalo_s (fast) or buffalo_l (accurate)
    providers=["CPUExecutionProvider"],  # CPU or GPU
    det_size=(640, 640),                 # Detection size (larger = more accurate, slower)
    db_dir="db/embeddings",              # Embedding storage directory
    threshold=0.40,                      # Recognition similarity threshold
    unknown_percent_cutoff=30.0,         # Confidence cutoff for "Unknown"
)
```

### Performance Tuning

In [app.py](app.py#L81-L83) `VideoProcessor` class:

```python
self.infer_interval = 0.4  # Seconds between inference runs (adjust for performance)
```

- Lower values: More responsive, higher CPU usage
- Higher values: Better performance, less responsive

### Model Selection

- `buffalo_s`: Faster, suitable for real-time applications on CPU
- `buffalo_l`: More accurate, requires more processing power

## How It Works

### Face Enrollment

1. User uploads one or more images
2. The system detects faces in each image
3. Face embeddings (512-dimensional vectors) are extracted
4. Embeddings from multiple images are averaged for robustness
5. The average embedding is normalized and saved as `{name}.npy`

### Face Recognition

1. Video frames are captured from the webcam
2. Faces are detected using InsightFace's detection model
3. Embeddings are extracted for each detected face
4. Cosine similarity is computed between detected and known embeddings
5. If similarity exceeds the threshold, the face is recognized
6. Results are displayed with bounding boxes and confidence scores

### Similarity Scoring

- **Cosine Similarity**: Measures similarity between embeddings (-1 to 1)
- **Threshold**: Default 0.40 (configurable)
- **Confidence Percentage**: Converted using sigmoid function for display

## Troubleshooting

### Camera Not Working

- Ensure browser permissions allow camera access
- Check if another application is using the camera
- Try refreshing the browser page

### Low Recognition Accuracy

- Enroll with more images (3-5 recommended)
- Ensure good lighting conditions
- Use frontal face images for enrollment
- Increase detection size: `det_size=(640, 640)` → `det_size=(1024, 1024)`
- Use `buffalo_l` model for better accuracy

### Performance Issues

- Reduce detection size: `det_size=(640, 640)` → `det_size=(320, 320)`
- Increase inference interval: `self.infer_interval = 0.6`
- Reduce max_faces: `max_faces=2` → `max_faces=1`
- Use `buffalo_s` model instead of `buffalo_l`

### Module Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the face recognition models
- [Streamlit](https://streamlit.io/) for the web framework
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for WebRTC support

## Contact

For questions or support, please open an issue in the repository.
