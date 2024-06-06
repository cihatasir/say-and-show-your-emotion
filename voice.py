import keras
import numpy as np
import librosa

def load_model(path):
    model = keras.models.load_model(path)
    model.summary()
    return model

def make_predictions(model, file):
    data, sampling_rate = process_audio(file)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=0)
    x = np.expand_dims(x, axis=2)
    predictions = model.predict(x)
    return predictions

def process_audio(file):
    data, sampling_rate = librosa.load(file)
    return data, sampling_rate

def format_predictions(predictions):
    emotions = ["angry", "happy", "neutral", "fearful", "calm", "sad", "disgust"]
    results = []
    for i, emotion in enumerate(emotions):
        results.append({
            "name": emotion.capitalize(),
            "percentage": f"{(predictions[0][i] * 100):.2f}%"
        })
    return results

def analyze_emotion(voice):
    model = load_model('Emotion_Voice_Detection_Model.h5')
    predictions = make_predictions(model, voice)
    return format_predictions(predictions)
