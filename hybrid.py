import os
import sys
import librosa
import numpy as np
import pandas as pd
import joblib
from preprocess import preprocess_file
from features import extract_features
import warnings
# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, message=".*The tree method `gpu_hist` is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Falling back to prediction using DMatrix.*")
def predict_voice_attributes(audio_file_path):
    """
    Predict both gender and age group from an audio file using the trained models.
    
    Args:
        audio_file_path: Path to the audio file
    
    Returns:
        Dictionary containing gender and age predictions with their confidence scores
    """
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} does not exist")
        return None
    
    print(f"Processing audio file: {audio_file_path}")
    
    # Step 1: Preprocess the audio file (only need to do this once)
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
    
    try:
        # Step 2: Extract features once for both models
        y, sr = librosa.load(temp_output_path, sr=16000)
        features_dict = extract_features(y, sr)
        
        # Step 3: Predict gender
        gender_prediction = predict_gender(features_dict)
        
        # Step 4: Predict age
        age_prediction = predict_age(features_dict)
        
        # Step 5: Clean up temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        # Return combined results
        if gender_prediction and age_prediction:
            gender, gender_confidence = gender_prediction
            age, age_confidence = age_prediction
            
            # Map numerical labels to descriptive classes
            gender_label = "Male" if gender == 0 else "Female"
            age_label = "Young (20s)" if age == 0 else "Older (50s)"
            
            combined_label = f"{gender_label}, {age_label}"
            
            return {
                "gender": gender_label,
                "gender_confidence": gender_confidence,
                "age": age_label,
                "age_confidence": age_confidence,
                "combined_label": combined_label
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return None

def predict_gender(features_dict):
    """Extract gender-specific features and predict using gender model"""
    try:
        # Prepare gender features
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
        
        # Load gender model and scaler
        model_path = 'gender_model.joblib'
        scaler_path = 'gender_scaler.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Error: Gender model or scaler file not found")
            return None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Scale features and predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        confidence = probability[prediction]
        
        print(f"Gender Prediction: {'Male' if prediction == 0 else 'Female'}")
        print(f"Gender Confidence: {confidence:.2f}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error during gender prediction: {str(e)}")
        return None

def predict_age(features_dict):
    """Extract age-specific features and predict using age model"""
    try:
        # Prepare age-specific features
        features = []
        
        # Pitch features
        features.extend([
            features_dict['pitch_mean'],
            features_dict['pitch_std']
        ])
        
        # Selected MFCCs (first 5)
        for i in range(1, 6):
            features.append(features_dict[f'mfcc_mean_{i}'])
        for i in range(1, 6):
            features.append(features_dict[f'mfcc_std_{i}'])
        
        # Spectral shape
        features.extend([
            features_dict['centroid_mean'], 
            features_dict['centroid_std'], 
            features_dict['bandwidth_mean'], 
            features_dict['bandwidth_std'],
            features_dict['rolloff_mean'], 
            features_dict['rolloff_std']
        ])
        
        # Energy features
        features.extend([
            features_dict['rms_mean'],
            features_dict['rms_std']
        ])
        
        # Chroma std features
        for i in range(1, 13):
            features.append(features_dict[f'chroma_std_{i}'])
        
        # Mel features (first 30)
        for i in range(1, 31):
            features.append(features_dict[f'mel_mean_{i}'])
        for i in range(1, 31):
            features.append(features_dict[f'mel_std_{i}'])
        
        # Reshape for model input
        X = np.array(features).reshape(1, -1)
        
        # Load age model and scaler
        model_path = 'age_model.joblib'
        scaler_path = 'age_scaler.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Error: Age model or scaler file not found")
            return None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Scale features and predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        confidence = probability[prediction]
        
        print(f"Age Prediction: {'Young (20s)' if prediction == 0 else 'Older (50s)'}")
        print(f"Age Confidence: {confidence:.2f}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error during age prediction: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = predict_voice_attributes(audio_path)
    
    if result:
        print("\n=== Final Prediction ===")
        print(f"Gender: {result['gender']} (Confidence: {result['gender_confidence']:.2f})")
        print(f"Age Group: {result['age']} (Confidence: {result['age_confidence']:.2f})")
        print(f"Speaker Profile: {result['combined_label']}")
    else:
        print("Prediction failed")