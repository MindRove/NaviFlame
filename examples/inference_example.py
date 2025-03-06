import os
import sys
import json
import cv2
from threading import Event, Thread
from queue import Queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naviflame.inference import real_time_inference, show_image_for_prediction
from naviflame.utils import FilterTypes, BiquadMultiChan, send_output_to_socket

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppresses logs except errors


#config loading
def load_config():
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json")),
        os.path.abspath("config.json")
    ]
    config_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if config_path is None:
        raise FileNotFoundError("Configuration file not found in the expected locations.")

    with open(config_path, "r") as f:
        config = json.load(f)

    config_dir = os.path.dirname(config_path)
    for key in ["data_path", "feature_extractor_path", "mlp_model_path", "scaler_path", "gesture_image_path"]:
        if key in config:
            config[key] = os.path.abspath(os.path.join(config_dir, config[key]))

    return config


def main():
    config = load_config()

    # Extract paths
    feature_extractor_path = config["feature_extractor_path"]
    mlp_model_path = config["mlp_model_path"]
    scaler_path = config["scaler_path"]
    gesture_image_path = config["gesture_image_path"]

    # Extract settings
    show_predicted_image = config["show_predicted_image"]
    send_to_socket = config["send_to_socket"]
    
    sampling_rate = 500
    model_input_len = 100

    # Define filters
    filters = [
        BiquadMultiChan(8, FilterTypes.bq_type_highpass, 4.5 / sampling_rate, 0.5, 0.0),
        BiquadMultiChan(8, FilterTypes.bq_type_notch, 50.0 / sampling_rate, 4.0, 0.0),
        BiquadMultiChan(8, FilterTypes.bq_type_lowpass, 100.0 / sampling_rate, 0.5, 0.0),
    ]

    print("Starting Real-Time Inference...")
    
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
            print(f"Predicted gesture: {prediction}")

            if send_to_socket:
                output_queue.put(prediction)

            if show_predicted_image:
                show_image_for_prediction(prediction, gesture_image_path, [])

    except KeyboardInterrupt:
        print("Inference stopped.")
        if send_to_socket:
            stop_event.set()
            socket_thread.join()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
