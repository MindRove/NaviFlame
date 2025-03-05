import os
import sys
import pickle
from threading import Event, Thread
from queue import Queue
import cv2
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from record import record_gestures
from fine_tune import fine_tune_model
from inference import real_time_inference, show_image_for_prediction
from utils import FilterTypes, BiquadMultiChan, BiquadMultiChan, FilterTypes, send_output_to_socket

# Load config
def load_config():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    json_path = os.path.join(base_dir, "config.json")
    with open(json_path, "r") as f:
        return json.load(f)

def main():
    config = load_config()
    # Paths from config
    data_path = config["data_path"]
    feature_extractor_path = config["feature_extractor_path"]
    mlp_model_path = config["mlp_model_path"]
    scaler_path = config["scaler_path"]
    gesture_image_path = config["gesture_image_path"]

    # Flags 
    record = config["record"]
    fine_tune = config["fine_tune"]
    show_predicted_image = config["show_predicted_image"]
    send_to_socket = config["send_to_socket"]
    
    # Skip gestures
    skip_gestures = []
    # Sampling rate and model input length
    sampling_rate = 500
    model_input_len = 100

    # Define filters
    filters = [
        BiquadMultiChan(8, FilterTypes.bq_type_highpass, 4.5 / sampling_rate, 0.5, 0.0), # Dc filter
        BiquadMultiChan(8, FilterTypes.bq_type_notch, 50.0 / sampling_rate, 4.0, 0.0), # 50 Hz noise
        BiquadMultiChan(8, FilterTypes.bq_type_lowpass, 100.0 / sampling_rate, 0.5, 0.0), 
    ]

    # Step 1: Record Gestures
    print("Step 1: Recording Gestures")
    if record:
        record_gestures(
            filters=filters,
            data_path=data_path,
            gesture_image_path=gesture_image_path,
            skip_gestures=skip_gestures,
            gestures_repeat=1,
            recording_time_sec=8,
            sampling_rate=sampling_rate,
            model_input_len=model_input_len,
            overlap_frac=model_input_len // 10,
        )
    else:
        print("Skipping gesture recording.")

    # Step 2: Fine-Tune the Model
    print("Step 2: Fine-Tuning the Model")
    
    if fine_tune:
        with open(data_path, "rb") as f:
            recorded_data, recorded_labels = pickle.load(f)

        scaler, val_accuracy = fine_tune_model(
            feature_extractor_path=feature_extractor_path,
            recorded_data=recorded_data,
            recorded_labels=recorded_labels,
            scaler_path=scaler_path,
            mlp_model_path=mlp_model_path,
        )
        print(f"Fine-tuning complete. Validation accuracy: {val_accuracy:.2f}")
    else:
        print("Skipping fine-tuning.")

    # Step 3: Real-Time Inference
    print("Step 3: Starting Real-Time Inference")
    if send_to_socket:
        stop_event = Event()
        output_queue = Queue()
        socket_thread = Thread(target=send_output_to_socket, args=(stop_event, output_queue))
        socket_thread.start()
    try:
        for prediction, probabilities in real_time_inference(
            feature_extractor_path=feature_extractor_path,
            mlp_model_path=mlp_model_path,
            scaler_path=scaler_path,
            filters=filters,
            model_input_len=model_input_len,
            gyro_threshold=500,
            prediction_threshold=0.4,
            batch_size=5,
        ):
            #print(f"Prediction: {prediction}, Probabilities: {probabilities.round(4)}")
            print(f"Predicted gesture: {prediction}")
            if send_to_socket:
                output_queue.put(prediction)
            if show_predicted_image:
                show_image_for_prediction(prediction, gesture_image_path, skip_gestures)
            
    except KeyboardInterrupt:
        print("Real-time inference stopped.")
        if send_to_socket:
            stop_event.set()
            socket_thread.join()
        # close all windows
        cv2.destroyAllWindows()

    print("Example script completed.")

if __name__ == "__main__":
    main()
