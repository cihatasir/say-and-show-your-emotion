import os
import uuid
from flask import Flask, flash, jsonify, redirect, render_template, Response, request
import cv2
import librosa
import numpy as np
import keras
from keras.models import model_from_json
import os
from voice import analyze_emotion, load_model
import sounddevice as sd
import soundfile as sf

app = Flask(__name__, static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

camera = cv2.VideoCapture(0)

json_file = open("emotion_recognition_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json, {'Sequential': keras.Sequential})
model.load_weights("emotion_recognition_model.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_frame = gray[y:y+h, x:x+w]
                face_frame = cv2.resize(face_frame, (48, 48))
                img = extract_features(face_frame)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, '%s' % (prediction_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

audio_frames = []
sample_rate = 44100
stream = None

def audio_callback(indata, frames, time, status):
    audio_frames.append(indata.copy())

@app.route('/start_record', methods=['POST'])
def start_record():
    global stream, audio_frames
    audio_frames = []
    if stream is None:
        stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
    stream.start()
    return jsonify({'message': 'Recording started'}), 200

@app.route('/stop_record', methods=['POST'])
def stop_record():
    global stream, audio_frames
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
    audio_data = np.concatenate(audio_frames)
    output_path = 'output.wav'
    sf.write(output_path, audio_data, sample_rate)
    
    emotion_result = analyze_emotion(output_path)
    print(emotion_result)
    os.remove(output_path)

    return jsonify({'message': 'Recording stopped and saved', 'emotion': emotion_result}), 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)
