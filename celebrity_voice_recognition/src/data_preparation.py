import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from feature_extraction import extract_features, synthesize_ai_voice

def prepare_dataset(real_voice_directory):
    features = []
    labels = []

    # Process real voices
    for filename in os.listdir(real_voice_directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(real_voice_directory, filename)
            audio, sr = librosa.load(file_path, sr=None)
            
            feature = extract_features(audio, sr)
            features.append(feature)
            labels.append(1)  # 1 for real voice

            # Create synthetic AI voice and add to dataset
            ai_audio = synthesize_ai_voice(audio, sr)
            ai_feature = extract_features(ai_audio, sr)
            features.append(ai_feature)
            labels.append(0)  # 0 for AI voice

    features = np.array(features)
    labels = np.array(labels)

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels, test_size=0.2, random_state=42)

    # Create 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the dataset and scaler
    joblib.dump((X_train, X_test, y_train, y_test), 'models/voice_dataset.pkl')
    joblib.dump(scaler, 'models/voice_feature_scaler.joblib')

    return X_train, X_test, y_train, y_test, scaler