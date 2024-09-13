import numpy as np
import sounddevice as sd
import queue
import threading
from feature_extraction import extract_features

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
PROCESS_INTERVAL = 3  # Process every 3 seconds of audio

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def predict_voice_type(model, scaler, audio):
    features = extract_features(audio, SAMPLE_RATE)
    features_normalized = scaler.transform([features])
    probabilities = model.predict_proba(features_normalized)[0]
    prediction = model.predict(features_normalized)[0]
    
    if prediction == 1:
        return "Real Voice", probabilities[1]
    else:
        return "Potential AI Voice", probabilities[0]

def process_audio(model, scaler):
    while True:
        audio_buffer = np.zeros((0, 1))
        while len(audio_buffer) < SAMPLE_RATE * PROCESS_INTERVAL:
            audio_buffer = np.vstack((audio_buffer, audio_queue.get()))
        
        audio = audio_buffer.flatten()
        voice_type, confidence = predict_voice_type(model, scaler, audio)
        print(f"Detected: {voice_type} (Confidence: {confidence:.2f})")

def start_real_time_recognition(model, scaler):
    global audio_queue
    audio_queue = queue.Queue()

    processing_thread = threading.Thread(target=process_audio, args=(model, scaler))
    processing_thread.start()

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopped listening.")

    processing_thread.join()