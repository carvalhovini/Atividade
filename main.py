import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import sys

# Tentar carregar o classificador Haar Cascade para detecção de rostos
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except cv2.error as e:
    print("Erro ao carregar o classificador Haar Cascade:", e)
    sys.exit(1)

# Tentar carregar o modelo de reconhecimento de emoções
try:
    emotion_model = load_model('emotion_detection_model.h5')
except OSError as e:
    print("Erro ao carregar o modelo de reconhecimento de emoções:", e)
    sys.exit(1)

emotion_labels = {0: 'Raiva', 1: 'Desgosto', 2: 'Medo', 3: 'Felicidade', 4: 'Tristeza', 5: 'Surpresa', 6: 'Neutro'}

def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=3)
    predicted_class = np.argmax(emotion_model.predict(face))
    return emotion_labels[predicted_class]

def start_emotion_detection(source, label_video, textbox):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erro ao abrir a fonte de vídeo.")
        return

    start_time = datetime.now()

    def update_frame():
        nonlocal cap
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostos no frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Processar cada rosto detectado
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Calcular a área do rosto
                face_area = w * h

                # Definir uma confiança mínima para aceitar a detecção
                min_confidence = 5000  # Ajuste conforme necessário

                if face_area > min_confidence:
                    emotion = predict_emotion(face_roi)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 0, 255), -1)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    timestamp = datetime.now() - start_time
                    textbox.insert(tk.END, f'{timestamp}: {emotion}\n')
                    textbox.see(tk.END)

            frame = cv2.resize(frame, (640, 400))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            label_video.configure(image=frame)
            label_video.image = frame

            label_video.after(30, update_frame)
        else:
            cap.release()

    update_frame()

def select_source(label_video, textbox):
    source = filedialog.askopenfilename(parent=root, filetypes=[("Video files", ".mp4;.avi;*.mov")])
    if source:
        start_emotion_detection(source, label_video, textbox)

def select_webcam(label_video, textbox):
    start_emotion_detection(0, label_video, textbox)

root = tk.Tk()
root.title("Detecção de Emoções")

video_label = ttk.Label(root)
video_label.pack()

textbox = tk.Text(root, width=50, height=10)
textbox.pack()

select_video_button = ttk.Button(root, text="Selecionar Vídeo", command=lambda: select_source(video_label, textbox))
select_video_button.pack(pady=5)

select_webcam_button = ttk.Button(root, text="Usar Webcam", command=lambda: select_webcam(video_label, textbox))
select_webcam_button.pack(pady=5)

quit_button = ttk.Button(root, text="Sair", command=root.quit)
quit_button.pack(pady=5)

root.mainloop()