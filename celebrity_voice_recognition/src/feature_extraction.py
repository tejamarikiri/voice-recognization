import librosa
import numpy as np
from scipy.stats import kurtosis, skew

def extract_features(audio, sr):
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)])
    
    # Harmonic features
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_ratio = np.sum(harmonic) / np.sum(percussive)
    
    # Statistical features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    kurtosis_value = kurtosis(audio)
    skew_value = skew(audio)
    
    # Combine all features
    features = np.hstack([
        mfcc_mean,
        mfcc_var,
        np.mean(spectral_centroids),
        np.mean(spectral_bandwidth),
        np.mean(spectral_rolloff),
        tempo,
        pitch_mean,
        harmonic_ratio,
        np.mean(zero_crossing_rate),
        kurtosis_value,
        skew_value
    ])
    
    return features

def synthesize_ai_voice(audio, sr):
    # Apply pitch shifting
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    
    # Apply time stretching
    stretched = librosa.effects.time_stretch(shifted, rate=1.2)
    
    # Apply some filtering
    y_harmonic, y_percussive = librosa.effects.hpss(stretched)
    
    # Combine and normalize
    synthetic = y_harmonic * 0.8 + y_percussive * 0.2
    synthetic = librosa.util.normalize(synthetic)
    
    return synthetic