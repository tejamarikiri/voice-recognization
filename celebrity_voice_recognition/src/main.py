import os
from data_preparation import prepare_dataset
from model_training import train_model
from real_time_recognition import start_real_time_recognition
import joblib

def main():
    # Set the path to your real voice samples
    real_voice_directory = 'D:\Projects\celebrity_voice_recognition\data\Real_voice'

    # Prepare the dataset
    X_train, X_test, paths_train, paths_test, scaler = prepare_dataset(real_voice_directory)

    # Train the model
    model = train_model(X_train, X_test, paths_train, paths_test)

    # Start real-time recognition
    start_real_time_recognition(model, scaler)

if __name__ == "__main__":
    main()