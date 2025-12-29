# Face Recognition System

A real-time face recognition application built with Python, InsightFace, and Streamlit. This project provides a web-based interface for enrolling users and recognizing faces through a webcam stream with high accuracy and performance.

## Features

- **Real-time Face Recognition**: Detect and recognize faces in real-time using webcam with optimized performance
- **User Enrollment**: Easy enrollment system with multi-image support for better accuracy
- **Advanced Face Detection**: Powered by InsightFace's state-of-the-art detection models (buffalo_s/buffalo_l)
- **Web Interface**: User-friendly Streamlit-based web interface with live video streaming
- **Similarity Scoring**: Display confidence percentages with sigmoid transformation for intuitive results
- **Multi-face Detection**: Support for detecting and recognizing multiple faces simultaneously
- **Persistent Storage**: Save and load face embeddings as NumPy arrays from disk
- **Optimized Performance**: Async processing, throttled inference, and EMA smoothing for smooth video streaming
- **Configurable Thresholds**: Adjustable recognition thresholds and cutoff parameters

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
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE.md                  # MIT License
├── assets/
│   └── icons/                  # UI assets (emotion icons if used)
├── db/
│   └── embeddings/             # Stored face embeddings (.npy files)
├── models/                     # InsightFace models (auto-downloaded on first run)
├── notebooks/
│   └── Emotion.ipynb           # Jupyter notebook for experimentation
└── src/
    ├── __init__.py
    ├── emotion_ui.py           # UI utilities for emotion/badge display
    ├── face_engine.py          # Core face recognition engine with FaceAnalysis
    ├── model/
    │   ├── __init__.py
    │   └── recognition_result.py  # Recognition result data model
    └── utils/
        ├── __init__.py
        └── util.py             # Utility functions (l2norm, cosine similarity, sigmoid)
```

## Installation

### Prerequisites

- Python 3.10 or higher (tested with Python 3.10 and 3.12)
- Webcam for real-time face recognition
- Modern web browser with WebRTC support (Chrome, Firefox, Edge recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd face-emotion
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using uv (recommended)
   uv venv uv_venv --python 3.10
   source uv_venv/bin/activate  # On Windows: uv_venv\Scripts\activate
   
   # Or using standard venv
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Using uv (faster)
   uv pip install -r requirements.txt
   
   # Or using standard pip
   pip install -r requirements.txt
   ```

   The key dependencies include:
   - `insightface==0.7.3`: Face recognition and detection
   - `streamlit`: Web interface framework
   - `opencv-python`: Image processing and computer vision
   - `onnxruntime`: Optimized model inference
   - `streamlit-webrtc==0.48.0`: Real-time WebRTC video streaming
   - `numpy`: Numerical computations for embeddings

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
2. The system will automatically detect and recognize faces in real-time
3. Recognized faces will display:
   - Green bounding box around the face
   - Name of the recognized person
   - Confidence percentage (based on cosine similarity)
4. Unknown faces (below threshold) will show "Unknown" with low confidence
5. Click "Stop" to end the video stream

### Managing the Database

- **Reload Database**: Click "Reload DB" in the sidebar to refresh the list of known users
- **View Known Users**: The sidebar displays all enrolled users
- **Manual Management**: Face embeddings are stored as `.npy` files in `db/embeddings/`
  - Delete files to remove users
  - The filename (without extension) is the user's name

## Configuration

### Face Recognition Engine

In [app.py](app.py#L17-L25), you can configure the `FaceEngine`:

```python
engine = FaceEngine(
    model_name="buffalo_s",              # Model: buffalo_s (fast) or buffalo_l (accurate)
    providers=["CPUExecutionProvider"],  # CPU or GPU (use "CUDAExecutionProvider" for GPU)
    det_size=(640, 640),                 # Detection size (larger = more accurate, slower)
    db_dir="db/embeddings",              # Embedding storage directory
    threshold=0.40,                      # Recognition similarity threshold (0.0-1.0)
    unknown_percent_cutoff=30.0,         # Confidence cutoff percentage for "Unknown"
)
```

### Performance Tuning

In [app.py](app.py#L81-L83) `VideoProcessor` class:

```python
self.infer_interval = 0.4  # Seconds between inference runs (adjust for performance)
```

- **Lower values (0.2-0.3)**: More responsive, higher CPU usage, better real-time tracking
- **Higher values (0.5-1.0)**: Better performance, less CPU usage, slightly delayed updates

### Model Selection

- **`buffalo_s`**: Faster inference, suitable for real-time applications on CPU, good balance of speed and accuracy
- **`buffalo_l`**: Higher accuracy, requires more processing power, recommended for GPU or when accuracy is critical

Models are automatically downloaded from InsightFace on first run and cached in the `models/` directory.

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

- **Cosine Similarity**: Measures similarity between face embeddings (range: -1 to 1)
  - Values closer to 1 indicate higher similarity
  - Typical good matches: 0.4-0.8
- **Threshold**: Default 0.40 (configurable in FaceEngine)
  - Matches above threshold are recognized
  - Matches below threshold show as "Unknown"
- **Confidence Percentage**: Converted using sigmoid function for intuitive display (0-100%)
  - Provides user-friendly visualization of match quality

## Troubleshooting

### Camera Not Working

- **Browser Permissions**: Ensure browser permissions allow camera access (check browser settings)
- **Camera in Use**: Check if another application is using the camera (close Zoom, Teams, etc.)
- **Refresh Browser**: Try refreshing the browser page or restarting the Streamlit server
- **HTTPS Required**: Some browsers require HTTPS for camera access in production
- **Check Console**: Look for errors in browser developer console (F12)

### Low Recognition Accuracy

- **Enroll More Images**: Use 3-5 images per person for better accuracy
- **Good Lighting**: Ensure good lighting conditions during enrollment and recognition
- **Frontal Faces**: Use frontal face images for enrollment (avoid extreme angles)
- **Increase Detection Size**: `det_size=(640, 640)` → `det_size=(1024, 1024)` for better accuracy
- **Better Model**: Use `buffalo_l` model instead of `buffalo_s` for higher accuracy
- **Adjust Threshold**: Lower threshold value (e.g., `threshold=0.35`) for more lenient matching
- **Image Quality**: Use high-resolution, clear images for enrollment

### Performance Issues

- **Reduce Detection Size**: `det_size=(640, 640)` → `det_size=(320, 320)` or `(480, 480)`
- **Increase Inference Interval**: `self.infer_interval = 0.6` or higher for less frequent processing
- **Reduce Max Faces**: `max_faces=2` → `max_faces=1` to detect fewer faces
- **Faster Model**: Use `buffalo_s` model instead of `buffalo_l`
- **CPU Optimization**: Ensure ONNX Runtime is using optimized CPU execution
- **Close Other Apps**: Close unnecessary applications to free up system resources

### Module Import Errors

Ensure all dependencies are installed:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt
```

If you encounter ONNX Runtime issues:
```bash
pip install --upgrade onnxruntime
```

For InsightFace model download issues, check your internet connection and firewall settings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - State-of-the-art face recognition models and framework
- [Streamlit](https://streamlit.io/) - Modern web framework for data applications
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) - Real-time WebRTC video streaming for Streamlit
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine

## Further Reading

- [InsightFace Documentation](https://insightface.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Face Recognition Theory](https://en.wikipedia.org/wiki/Facial_recognition_system)

## Contact

For questions, suggestions, or support, please open an issue in the GitHub repository.
