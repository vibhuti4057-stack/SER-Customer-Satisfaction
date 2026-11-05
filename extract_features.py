import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

data = []
for root, dirs, files in os.walk('data'):
    for file in files:
        if file.endswith('.wav'):
            path = os.path.join(root, file)
            emotion_code = file.split('-')[2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            features = extract_features(path)
            data.append([*features, emotion])

columns = [f'feature_{i}' for i in range(len(data[0]) - 1)] + ['emotion']
df = pd.DataFrame(data, columns=columns)
df.to_csv('features.csv', index=False)
print("Done! features.csv created with", len(df), "rows")