import os
import sys
import librosa
import numpy as np
import pandas as pd
import joblib
from preprocess import preprocess_file
from features import extract_features

def predict_gender(audio_file_path):
    """
    Predict gender from an audio file using the trained model.
    
    Args:
        audio_file_path: Path to the audio file
    
    Returns:
        Prediction result ("Male" or "Female") and confidence
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} does not exist")
        return None
    
    print(f"Processing audio file: {audio_file_path}")
    
    # Step 1: Preprocess the audio file
    temp_output_path = "temp_processed_audio.wav"
    success = preprocess_file(
        (audio_file_path, temp_output_path),
        target_sr=16000,
        trim_silence=True,
        reduce_noise=False
    )
    
    if not success:
        print("Error: Failed to preprocess the audio file")
        return None
    
    # Step 2: Extract features
    try:
        y, sr = librosa.load(temp_output_path, sr=16000)
        features_dict = extract_features(y, sr)
        
        # Step 3: Prepare features in the right format
        features = []
        
        # Pitch
        features.append(features_dict['pitch_mean'])
        features.append(features_dict['pitch_std'])
        
        # MFCCs
        for i in range(1, 14):
            features.append(features_dict[f'mfcc_mean_{i}'])
        for i in range(1, 14):
            features.append(features_dict[f'mfcc_std_{i}'])
        
        # Chroma
        for i in range(1, 13):
            features.append(features_dict[f'chroma_mean_{i}'])
        for i in range(1, 13):
            features.append(features_dict[f'chroma_std_{i}'])
        
        # Spectral shape
        features.extend([
            features_dict['centroid_mean'], 
            features_dict['centroid_std'], 
            features_dict['bandwidth_mean'], 
            features_dict['bandwidth_std'],
            features_dict['rolloff_mean'], 
            features_dict['rolloff_std']
        ])
        
        # Energy / voicing
        features.extend([
            features_dict['rms_mean'], 
            features_dict['rms_std'], 
            features_dict['zcr_mean'], 
            features_dict['zcr_std']
        ])
        
        # Mel (only first 13)
        for i in range(1, 14):
            features.append(features_dict[f'mel_mean_{i}'])
        for i in range(1, 14):
            features.append(features_dict[f'mel_std_{i}'])
        
        # Reshape for model input
        X = np.array(features).reshape(1, -1)
        
        # Step 4: Load the model and scaler
        model_path = 'gender_model.joblib'
        scaler_path = 'gender_scaler.joblib'
        
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return None
            
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler file {scaler_path} not found")
            return None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Step 5: Scale features using the saved scaler
        X_scaled = scaler.transform(X)
        
        # Step 6: Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        # Return result
        result = "Male" if prediction == 0 else "Female"
        confidence = probability[prediction]
        
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}")
        
        return result, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = predict_gender(audio_path)
    
    if result:
        gender, confidence = result
        print(f"\nPredicted gender: {gender} (Confidence: {confidence:.2f})")
    else:
        print("Prediction failed")