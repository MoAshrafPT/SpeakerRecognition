import librosa
import numpy as np
import pandas as pd
import os
#from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


def extract_features(y, sr=16000):
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    pitches, _ = librosa.core.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_mean = np.mean(mel_spectrogram, axis=1)
    mel_std = np.std(mel_spectrogram, axis=1)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_std = np.std(spectral_rolloff)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    formants = extract_formants(y, sr)

    feature_dict = {}
    for i in range(len(mfcc_mean)):
        feature_dict[f'mfcc_mean_{i+1}'] = mfcc_mean[i]
    for i in range(len(mfcc_std)):
        feature_dict[f'mfcc_std_{i+1}'] = mfcc_std[i]
    for i in range(len(chroma_mean)):
        feature_dict[f'chroma_mean_{i+1}'] = chroma_mean[i]
    for i in range(len(chroma_std)):
        feature_dict[f'chroma_std_{i+1}'] = chroma_std[i]
    for i in range(len(mel_mean)):
        feature_dict[f'mel_mean_{i+1}'] = mel_mean[i]
    for i in range(len(mel_std)):
        feature_dict[f'mel_std_{i+1}'] = mel_std[i]

    for i, formant in enumerate(formants[:4], 1):  # First 4 formants
        feature_dict[f'formant_{i}'] = formant

    feature_dict.update({
        'centroid_mean': centroid_mean,
        'centroid_std': centroid_std,
        'bandwidth_mean': bandwidth_mean,
        'bandwidth_std': bandwidth_std,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'rolloff_mean': rolloff_mean,
        'rolloff_std': rolloff_std,
        'rms_mean': rms_mean,
        'rms_std': rms_std
    })

    return feature_dict


def extract_formants(y, sr):
    """
    Extract formant frequencies using LPC analysis.
    
    Args:
        y: Audio signal
        sr: Sample rate
    
    Returns:
        List of formant frequencies (in Hz)
    """
    # Pre-emphasis to amplify high frequencies (improves formant detection)
    y_emph = librosa.effects.preemphasis(y)
    
    # Frame the signal (25ms windows with 10ms hop)
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    frames = librosa.util.frame(y_emph, frame_length=frame_length, hop_length=hop_length)
    
    # Number of LPC coefficients (rule of thumb: sampling_rate / 1000 + 2)
    n_lpc = int(sr / 1000) + 2
    
    # Initialize array to store formants for each frame
    all_formants = []
    
    # Process each frame
    for frame in frames.T:
        # Apply window function
        frame = frame * np.hamming(len(frame))
        
        # LPC analysis
        a_lpc = librosa.lpc(frame, order=n_lpc)
        
        # Find roots of the LPC polynomial
        roots = np.roots(a_lpc)
        
        # Keep only roots with positive imaginary part (and inside unit circle)
        roots = roots[np.imag(roots) > 0]
        
        # Convert roots to frequencies in Hz
        angles = np.arctan2(np.imag(roots), np.real(roots))
        formants = angles * (sr / (2 * np.pi))
        
        # Sort formants by frequency
        formants = sorted(formants)
        
        # Store formants for this frame
        if len(formants) > 0:
            all_formants.append(formants)
    
    # If no formants were found, return zeros
    if not all_formants:
        return [0, 0, 0, 0]
    
    # Average formants across all frames
    # First, ensure all frames have the same number of formants
    max_formants = max(len(f) for f in all_formants)
    padded_formants = []
    for f in all_formants:
        if len(f) < max_formants:
            # Pad with zeros if fewer formants were found
            padded_formants.append(f + [0] * (max_formants - len(f)))
        else:
            padded_formants.append(f)
    
    # Convert to numpy array for easy averaging
    formant_array = np.array(padded_formants)
    
    # Calculate mean formants (up to first 4)
    mean_formants = np.mean(formant_array, axis=0)
    
    # Pad with zeros if fewer than 4 formants
    if len(mean_formants) < 4:
        mean_formants = np.append(mean_formants, [0] * (4 - len(mean_formants)))
    
    return mean_formants[:4]  # Return first 4 formants


def create_label_mapping(path='data/filtered_speech_data.csv'):
    label_df = pd.read_csv(path)
    label_df['filename'] = label_df['path'].apply(os.path.basename)
    return label_df[['filename', 'label']]


def process_batch(audio_files, label_df):
    batch_features = []
    batch_filenames = []

    for audio_file in audio_files:
        try:
            y, sr = librosa.load(audio_file, sr=16000)
            features = extract_features(y, sr)
            features['filename'] = os.path.basename(audio_file)
            batch_features.append(features)
            batch_filenames.append(os.path.basename(audio_file))
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")

    df = pd.DataFrame(batch_features)
    df = pd.merge(df, label_df, on='filename', how='left')
    return df


def run_batched_processing():
    preprocessed_audio_path = "clean_audio"
    output_path = "data/features.csv"
    #output_pca_path = "data/features_pca.csv"
    label_df = create_label_mapping('data/filtered_speech_data.csv')
    audio_files = librosa.util.find_files(preprocessed_audio_path, ext=['wav', 'mp3'])
    batch_size = 10000

    all_dfs = []
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Processing Batches"):
        batch_files = audio_files[i:i + batch_size]
        batch_df = process_batch(batch_files, label_df)
        all_dfs.append(batch_df)

    features_df = pd.concat(all_dfs, ignore_index=True)
    features_df.to_csv(output_path, index=False)
    print(f"Saved full features to {output_path}")

    feature_cols = features_df.columns.drop(['filename', 'label'])
    X = features_df[feature_cols].values

    #ipca = IncrementalPCA(n_components=min(20, X.shape[1]))
    #for i in range(0, len(X), batch_size):
    #    ipca.partial_fit(X[i:i + batch_size])

    #X_pca = np.concatenate([ipca.transform(X[i:i + batch_size]) for i in range(0, len(X), batch_size)], axis=0)
    #pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    #pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    #pca_df.insert(0, 'filename', features_df['filename'])
    #pca_df = pd.merge(pca_df, label_df, on='filename', how='left')
    #pca_df.to_csv(output_pca_path, index=False)
    #print(f"Saved PCA features to {output_pca_path}")
    #print(f"Total variance explained: {ipca.explained_variance_ratio_.sum():.2f}")


if __name__ == "__main__":
    run_batched_processing()
