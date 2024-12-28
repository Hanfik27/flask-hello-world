from flask import Flask, render_template, Response
import cv2
import random

app = Flask(__name__)

# Load Haarcascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize webcam
camera = cv2.VideoCapture(0)

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Deteksi mata
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        # Deteksi senyuman
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)

        # Hitung skor deteksi
        eye_count = len(eyes)
        smile_count = len(smiles)

        # Tentukan emosi dan persentase
        if smile_count > 0:
            emotion = "Happy"
            confidence = random.uniform(90, 100)  # Senyuman = 90-100% acak
        elif eye_count < 2:
            emotion = "Stressed"
            confidence = random.uniform(60, 80)  # Mata kurang jelas = 60-80% acak
        elif eye_count >= 2 and smile_count == 0:
            emotion = "Angry"
            confidence = random.uniform(75, 85)  # Mata jelas, tidak ada senyuman = 75-85% acak
        elif eye_count >= 2 and smile_count == 0:
            emotion = "Sad"
            confidence = random.uniform(60, 75)  # Mata jelas, tidak ada senyuman = 60-75% acak
        else:
            emotion = "Neutral"
            confidence = random.uniform(40, 60)

        # Gambar kotak di sekitar wajah dan tambahkan label emosi dan persentase
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
