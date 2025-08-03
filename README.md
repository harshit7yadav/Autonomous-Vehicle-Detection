# Real-time Autonomous Driving Computer Vision Demo

## Overview

This is a production-ready Streamlit web application that demonstrates computer vision techniques for autonomous driving with real-time camera support and video processing. The application processes dashcam images, videos, camera captures, and live camera feeds providing four key CV capabilities: lane detection, traffic sign classification, object detection, and steering angle prediction. It uses efficient computer vision implementations that simulate deep learning models for real-world autonomous driving perception systems with enhanced accuracy and reduced false positives.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with mobile-responsive design
- **Layout**: Wide layout with sidebar for controls and main area for visualization
- **Input Methods**: File uploader, video uploader, camera capture, real-time camera feed
- **Components**: Multi-modal input interface, real-time processing display, interactive visualizations using Plotly and Matplotlib

### Backend Architecture
- **Core Framework**: Python-based modular architecture
- **Model Layer**: Separate model classes for each CV task (lane detection, object detection, traffic sign classification, steering prediction)
- **Utility Layer**: Image processing and visualization utilities
- **Processing Pipeline**: Sequential processing of uploaded images through all CV models

### Enhanced Model Implementations
The application uses advanced computer vision techniques to simulate deep learning models with improved accuracy:
- **Lane Detection**: Simulates U-Net using Canny edge detection and Hough line transform
- **Object Detection**: Enhanced YOLOv8n simulation with multi-factor validation and reduced false positives
- **Traffic Sign Classification**: Improved ResNet18 simulation with strict color/shape filtering to eliminate people misclassification
- **Steering Prediction**: Simulates NVIDIA PilotNet using lane-based geometric calculations

## Key Components

### Model Classes
1. **LaneDetectionModel**: Processes images to detect lane markings using edge detection
2. **ObjectDetectionModel**: Detects vehicles, pedestrians, and traffic lights
3. **TrafficSignClassifier**: Identifies and classifies traffic signs
4. **SteeringPredictionModel**: Predicts steering angles based on lane positions

### Utility Modules
1. **Image Processing**: Handles preprocessing, resizing, and format conversion
2. **Visualization**: Creates overlays, bounding boxes, and steering visualizations

### Main Application
- **Streamlit Interface**: Provides web-based UI for image upload and results display
- **Model Caching**: Uses Streamlit's caching mechanism for model initialization
- **Interactive Visualizations**: Real-time display of CV results

## Data Flow

1. **Multi-modal Input**: User selects input method (upload image/video, camera capture, or real-time)
2. **Media Acquisition**: Image/video obtained from file upload, camera capture, or live camera feed
3. **Video Processing**: Videos are processed frame-by-frame with efficient sampling
4. **Preprocessing**: Images/frames are normalized and prepared for model input
5. **Model Inference**: Images/frames processed through all four CV models sequentially
6. **Real-time Processing**: Live camera feeds processed with optimized frame rate
7. **Visualization**: Results overlaid on original images/video frames with real-time annotations
8. **Interactive Analysis**: Users can view detailed results, timeline charts, confidence scores, and performance metrics
9. **Video Timeline**: Interactive charts showing steering angles and detections over time

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **PIL**: Image processing
- **Matplotlib**: Static plotting
- **Plotly**: Interactive visualizations

### Computer Vision Pipeline
- **Edge Detection**: Canny edge detector for lane detection
- **Line Detection**: Hough transform for lane line identification
- **Contour Analysis**: Object boundary detection
- **Color Filtering**: Traffic light and sign detection
- **Template Matching**: Shape-based object recognition

## Deployment Strategy

### Local Development
- Direct Python execution with Streamlit server
- All dependencies managed through pip/conda
- Models loaded once and cached for performance

### Production Considerations
- Implementation uses efficient computer vision algorithms optimized for real-time processing
- Memory usage is minimal due to lightweight CV operations
- Real-time camera processing ready for deployment on edge devices
- Mobile browser compatibility for camera access (iOS Safari, Android Chrome)
- Scalable architecture supports cloud deployment with load balancing
- Future enhancement would involve replacing CV implementations with actual trained neural networks

### Performance Optimization
- Model caching prevents reloading on each request
- Image preprocessing optimized for web display
- Efficient numpy operations for real-time processing

## Architecture Benefits

1. **Modular Design**: Each CV task is isolated in separate model classes
2. **Real-time Processing**: Live camera feed support with optimized performance
3. **Multi-modal Input**: Supports file upload, camera capture, and live streaming
4. **Deployment Ready**: Production-optimized for cloud and edge deployment
5. **Mobile Compatible**: Works on mobile browsers with camera access
6. **Extensibility**: Easy to replace CV implementations with real neural networks
7. **Interactive Demo**: Provides immediate visual feedback for educational purposes
8. **Lightweight**: Runs without GPU requirements or large model files
9. **Web-Based**: Accessible through browser without local installation

## Future Enhancements

The mock implementations can be replaced with actual pre-trained models:
- Replace lane detection with trained U-Net or SegNet
- Integrate YOLOv8 for real object detection
- Add ResNet or EfficientNet for traffic sign classification
- Implement NVIDIA PilotNet for steering prediction

The modular architecture supports these upgrades without changing the core application structure.

## Recent Changes

**August 3, 2025**: Enhanced with video processing and improved model accuracy
- Added video upload functionality (MP4, AVI, MOV, MKV support)
- Implemented comprehensive video processing pipeline with frame-by-frame analysis
- Created interactive video analysis with steering timeline and detection charts
- Significantly improved traffic sign detection accuracy to reduce false positives
- Enhanced object detection model with better filtering and validation
- Added detection timeline visualization showing objects and signs over time
- Improved model precision to distinguish between actual objects and background noise