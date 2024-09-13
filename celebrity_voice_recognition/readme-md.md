# AI Voice Detection System

This project implements a machine learning-based system to distinguish between real and AI-generated voices using Python. It uses a one-class SVM model trained on real voice samples to detect anomalies that could indicate AI-generated voices.

## Features

- Real-time voice analysis
- Feature extraction from audio samples using librosa
- One-class SVM model for anomaly detection
- Easy-to-use command-line interface

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-voice-detection.git
   cd ai-voice-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset:
   - Create a directory (e.g., `real_voices`) and place your .wav files of real voices in it.

2. Update the `main.py` file:
   - Set the `real_voice_directory` variable to the path of your dataset directory.

3. Run the main script:
   ```
   python main.py
   ```

   This will:
   - Prepare the dataset
   - Train the model
   - Start real-time voice detection

4. Speak into your microphone or play audio.
   The system will classify the input as either "Real Voice" or "Potential AI Voice".

## Project Structure

- `main.py`: Entry point of the application
- `data_preparation.py`: Handles dataset preparation and feature extraction
- `model_training.py`: Implements the one-class SVM model training
- `real_time_recognition.py`: Manages real-time audio processing and voice classification
- `feature_extraction.py`: Contains functions for extracting audio features

## Limitations

- The system is trained only on real voice samples, which may lead to false positives for unusual real voices.
- Very sophisticated AI-generated voices that closely mimic real voice characteristics might not be detected.

## Future Improvements

- Incorporate a dataset of AI-generated voices for more accurate classification
- Implement more advanced audio feature extraction techniques
- Explore deep learning models for improved performance

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
