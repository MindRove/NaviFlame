# naviFlame
MindRove's middle-sized AI system for detecting hand and finger motions from wearable sensor data.

naviFlame implements a pipeline for recording, fine-tuning, and performing real-time inference of gesture-based inputs using a MindRove device. The system incorporates signal processing, feature extraction, SVM-based classification, and real-time visualization.

## Features
- Record gestures with EMG data from the MindRove device.
- Fine-tune an SVM model for user.
- Real-time inference and display of gesture predictions.


---

## Table of Contents
1. [Setup](#setup)
2. [Configuration](#configuration)
3. [Usage](#usage)
4. [Input Parameters](#input-parameters)
5. [System Pipeline](#system-pipeline)
6. [Acknowledgements](#acknowledgements)

---

## Setup

### Prerequisites
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
    - feature_extractor_path: Path to the pretrained feature extractor model.
    - svm_model_path: Path to save the trained SVM model.
    - scaler_path: Path to save the scaler used in feature scaling.
    - gesture_image_path: Directory containing gesture images.

- Flags:
    - record: Enables/disables gesture recording.
    - fine_tune: Enables/disables SVM fine-tuning.
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
### 2. Fine-Tune SVM Model
Set fine_tune to True in example.py. After recording gestures, the script will fine-tune the SVM model.
### 3. Real-Time Inference
Ensure recorded data exists and the SVM model is fine-tuned. Run the script to perform real-time inference.

## Input Parameters

### Recording Parameters
| Parameter             | Description                                      | Default Value             |
|-----------------------|--------------------------------------------------|---------------------------|
| `filters`             | List of signal filters applied during recording. | Highpass, Notch, Lowpass  |
| `data_path`           | Path to save gesture data.                       | `data/recorded_gestures.pkl` |
| `gesture_image_path`  | Directory containing gesture images.             | `gestures/`               |
| `skip_gestures`       | List of gesture IDs to skip.                     | `[]`                      |
| `gestures_repeat`     | Number of repetitions for each gesture.          | `1`                       |
| `recording_time_sec`  | Duration for recording each gesture.             | `8 seconds`               |

### Fine-Tuning Parameters
| Parameter             | Description                                      | Default Value             |
|-----------------------|--------------------------------------------------|---------------------------|
| `C`                   | SVM regularization parameter.                   | `5.0`                     |
| `kernel`              | SVM kernel type.                                | `rbf`                     |
| `gamma`               | Kernel coefficient for SVM.                     | `scale`                   |

### Inference Parameters
| Parameter             | Description                                      | Default Value             |
|-----------------------|--------------------------------------------------|---------------------------|
| `model_input_len`     | Input length for the gesture model.              | `100` samples             |
| `gyro_threshold`      | Threshold for gyro data filtering.               | `2000`                    |
| `prediction_threshold`| Minimum confidence for gesture predictions.      | `0.4`                     |
| `batch_size`          | Number of samples processed per batch.           | `5`                       |



## System Pipeline
1. Data Recording:

The record_gestures function records EMG data from the MindRove board.
Signals are preprocessed using filters for noise reduction.
2. Fine-Tuning:

The fine_tune_svm function extracts features and trains an SVM classifier.
Data is scaled, and model performance is validated.
3. Real-Time Inference:

The real_time_inference function processes live EMG data and predicts gestures.
Predictions are visualized and optionally sent via a socket connection.
