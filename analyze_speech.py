import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import librosa
import numpy as np
import joblib

model = joblib.load('saved_model/emotion_model.pkl')
analyzer = SentimentIntensityAnalyzer()

def extract_features(file_path):
    audio, sr_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr_rate).T, axis=0)
    return np.hstack([mfcc, chroma, mel]).reshape(1, -1)

emotion_score_map = {
    'happy': 1.0, 'calm': 0.8, 'neutral': 0.5,
    'surprised': 0.6, 'sad': 0.2, 'angry': 0.0,
    'fearful': 0.1, 'disgust': 0.1
}

def get_satisfaction(audio_file):
    features = extract_features(audio_file)
    emotion = model.predict(features)[0]
    emotion_score = emotion_score_map.get(emotion, 0.5)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except:
        text = "(could not understand audio)"

    sentiment = analyzer.polarity_scores(text)
    sentiment_score = (sentiment['compound'] + 1) / 2

    final_score = round(0.6 * emotion_score + 0.4 * sentiment_score, 2)

    return {
        'emotion': emotion,
        'text': text,
        'sentiment_score': round(sentiment_score, 2),
        'satisfaction_score': final_score
    }