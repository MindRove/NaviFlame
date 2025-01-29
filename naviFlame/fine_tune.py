from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pickle
import logging
import numpy as np
import sys
from keras.models import load_model
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NaviFlame.utils import MyMagnWarping, MyScaling
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import time


def fine_tune_model(
    feature_extractor_path,
    recorded_data, 
    recorded_labels, 
    mlp_model_path,
    scaler_path,
):
    """
    Fine-tunes the MLP model using the recorded data.
    
    Args:
        model (keras.models.Model): Feature extractor model.
        recorded_data (list): List of recorded data.
        recorded_labels (list): List of recorded labels.
        mlp_model_path (str): Path to save the MLP model.
        scaler_path (str): Path to save the scaler.
        
        Returns:
            tuple: (MLP model, Scaler, Validation accuracy)        
    """
    model = load_model(feature_extractor_path, custom_objects={"MyMagnWarping": MyMagnWarping, "MyScaling": MyScaling})

    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split( np.array(recorded_data), np.array(recorded_labels), test_size=0.2, random_state=42)

    # Feature extraction
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer("dense_8").output)
    features_train = feature_extractor.predict(X_train)
    features_val = feature_extractor.predict(X_val)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_train)
    X_val_scaled = scaler.transform(features_val)
    logging.info("Data preprocessed and scaled.")

    # Save the scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        logging.info("Scaler saved.")

    # MLP model training
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), 
        max_iter=2000, 
        random_state=42,
        activation='tanh',  
        solver='lbfgs',  
        alpha=0.01,
        learning_rate='constant',
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_accuracy = mlp.score(X_val_scaled, y_val)
    logging.info(f"MLP validation accuracy: {mlp_accuracy}")
    logging.info(f"MLP cross-validation accuracy: {cross_val_score(mlp, X_train_scaled, y_train, cv=5)}")

    # Save the MLP model
    with open(mlp_model_path, "wb") as f:
        pickle.dump(mlp, f)
        logging.info("MLP model saved.")

    if mlp_accuracy < 0.60:
        response = input(
            "MLP validation accuracy is below 60%. Would you like to stop and check the device (Yes/No)? "
        ).strip().lower()
        if response in ("yes", "y"):
            logging.info(
                "Suggested actions: \n- Check device connection.\n- Ensure proper device positioning.\n- Clean the device sensors.\n- Record new data."
            )
            logging.info("Exiting the code.")
            sys.exit(0)  # Terminates the program.
        else:
            logging.info("Continuing despite low validation accuracy.")

    return scaler, mlp_accuracy
