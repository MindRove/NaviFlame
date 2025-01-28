# NaviFlame
[MindRove](https://mindrove.com/)'s middle-sized AI system for detecting hand and finger motions from wearable sensor data.

The NaviFlame implements a pipeline for recording, fine-tuning, and performing real-time inference of gesture-based inputs using a MindRove device. The system incorporates signal processing, Deep Learning based feature extraction, SVM/MLP-based classification, and real-time visualization.

## Features
- Record gestures with EMG data from the MindRove device.
- Fine-tune a hybrid Deep Learning plus SVM/MLP model for the user (only the SVM/MLP part has to be fine-tuned making the process super quick)
- Real-time AI inference and display of gesture predictions.
- A Unity-based application that visualizes data sent from `NaviFlame` via a socket.


---

## Table of Contents
1. [Setup](#setup)
2. [Configuration](#configuration)
3. [Usage](#usage)
4. [Input Parameters](#input-parameters)
5. [System Pipeline](#system-pipeline)
6. [Unity Visualization](#unity-visualization)
7. [Contact](#contact)

---

## Setup

### Requirements
- Python >=3.7 and <3.11.
- Libraries:
  - `tensorflow==2.12.0`
  - `scikit-learn`
  - `opencv-python`
  - `numpy`
  - `mindrove` (MindRove SDK)

### Installation
1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Ensure the MindRove device is properly placed and connected.


## Configuration
All configuration settings are defined in the example.py file. Below are key variables to modify based on your requirements:

### Key Configuration Variables:
- Paths:
    - data_path: Path to save recorded gesture data.
    - feature_extractor_path: Path to the pretrained feature extractor AI model.
    - svm_model_path: Path to save the trained SVM model.
    - mlp_model_path: Path to save the trained MLP model.
    - scaler_path: Path to save the scaler used in feature scaling.
    - gesture_image_path: Directory containing gesture images.

- Flags:
    - record: Enables/disables gesture recording.
    - fine_tune: Enables/disables fine-tuning.
    - show_predicted_image: Enables displaying gesture images during inference.
    - send_to_socket: Sends predictions to a socket server.

- Data Parameters:
    - sampling_rate: Sampling rate for EMG data (default: 500 Hz).
    - model_input_len: Input length for gesture model (default: 100 samples).
    - filters: List of filters applied to preprocess EMG signals.
    
    
## Usage
```python example.py```
### 1. Record Gestures
Run the example.py script with the record flag set to True to capture gestures.
### 2. Fine-Tune SVM/MLP Model
Set fine_tune to True in example.py. After recording gestures, the script will fine-tune the SVM/MLP model.
### 3. Real-Time Inference
Ensure recorded data exists and the model is fine-tuned. Run the script to perform real-time inference.

## Input Parameters

### Recording Parameters
| Parameter             | Description                                      | Default Value             |
|-----------------------|--------------------------------------------------|---------------------------|
| `filters`             | List of filters applied during recording.        | - |
| `data_path`           | Path to save the recorded data.                  | - |
| `gesture_image_path`  | Path to the gesture images.                      | `gestures/`               |
| `skip_gestures`       | List of gesture IDs to skip.                     | `[]`                      |
| `gestures_repeat`     | Number of repetitions for each gesture.          | `1`                       |
| `recording_time_sec`  | Duration to record each gesture.                 | `8` seconds               |
| `sampling_rate`       | Sampling rate of the MindRove board.             | `500` Hz                  |
| `model_input_len`     | Length of input data to the model.               | `100` samples             |
| `overlap_frac`        | Overlap fraction between samples.                | `10`                      |

### Fine-Tuning Parameters
| Parameter             | Description                                      | Default Value             |
|-----------------------|--------------------------------------------------|---------------------------|
| `feature_extractor_path` | Path to the feature extractor model.           | -    |
| `recorded_data`       | List of recorded data used for training.         | -                      |
| `recorded_labels`     | List of labels for the recorded data.            | -                       |
| `svm_path`            | Path to save the trained SVM model.              | -      |
| `mlp_model_path`      | Path to save the trained MLP model.              | -      |
| `scaler_path`         | Path to save the scaler used for normalization.  | -         |
| `C`                   | SVM regularization parameter.                   | `5.0`                     |
| `kernel`              | SVM kernel type.                                | `rbf`                     |
| `gamma`               | Kernel coefficient for SVM.                     | `scale`                   |

### Inference Parameters
| Parameter             | Description                                      | Default Value             |
|-----------------------|--------------------------------------------------|---------------------------|
| `feature_extractor_path` | Path to the feature extractor model.           | -    |
| `svm_model_path`      | Path to the SVM model.                           | -      |
| `mlp_model_path`      | Path to the MLP model.                           | -      |
| `scaler_path`         | Path to the scaler used for normalization.       | -         |
| `filters`             | List of filters applied during inference.        | -  |
| `model_input_len`     | Length of input data to the model.               | `100` samples             |
| `gyro_threshold`      | Threshold for gyro data filtering.               | `2000`                    |
| `prediction_threshold`| Confidence threshold for gesture predictions.    | `0.4`                     |
| `batch_size`          | Number of samples processed for one prediction.  | `5`                       |



## System Pipeline
1. Data Recording:

- The record_gestures function records EMG data from the MindRove board.
- Signals are preprocessed using filters for noise reduction.
2. Fine-Tuning:

- The fine_tune_svm function extracts features and trains an SVM classifier.
- Data is scaled, and model performance is validated.
3. Real-Time Inference:

- The real_time_inference function processes live EMG data and predicts gestures.
- Predictions are visualized and optionally sent via a socket connection.


---

## Unity Visualization

### Description
The Unity application is designed to visualize the gestures detected by the `NaviFlame` system. It connects to the same socket used for data transmission and provides a real-time display of the gestures as 3D animations.

### Running the Unity Application
1. Decompress the zip file into a folder of your choice (e.g., NaviFlame_unity_visualizer/). 
2. Inside the decompressed folder, locate the NaviFlame_visualizer.exe file and run it to start the Unity application.
3. Ensure that the Unity application is run prior to the  NaviFlame. The visualizer will automatically connect to the socket and display the gestures real-time.

---
## Contact
For support, collaboration, or queries, please reach out via:
- **Email**: [support@mindrove.com](mailto:info@mindrove.com)
- **Website**: [MindRove](https://mindrove.com/)

