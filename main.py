import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    face_detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
except cv2.error as e:
    print("Erro ao carregar o detector de rosto:", e)
    sys.exit(1)

try:
    emotion_model = load_model('emotion_detection_model.h5')
except OSError as e:
    print("Erro ao carregar o modelo de reconhecimento de emoções:", e)
    sys.exit(1)

emotion_labels = {0: 'Raiva', 1: 'Desgosto', 2: 'Medo', 3: 'Felicidade', 4: 'Tristeza', 5: 'Surpresa', 6: 'Neutro'}
emotion_counts = {emotion: 0 for emotion in emotion_labels.values()}

def update_graphs():
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    ax.clear()
    ax.bar(emotions, counts, color='skyblue')
    ax.set_title('Contagem de Emoções Detectadas')
    ax.set_xlabel('Emoções')
    ax.set_ylabel('Contagem')
    canvas.draw()

def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=3)
    predicted_class = np.argmax(emotion_model.predict(face))
    return emotion_labels[predicted_class]

def start_emotion_detection(source, label_video):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erro ao abrir a fonte de vídeo.")
        return

    def update_frame():
        nonlocal cap
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostos na imagem
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Ajuste o limiar de confiança conforme necessário
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    face_roi = frame[startY:endY, startX:endX]
                    emotion = predict_emotion(face_roi)
                    emotion_counts[emotion] += 1
                    update_graphs()
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    cv2.rectangle(frame, (startX, startY - 40), (endX, startY), (0, 0, 255), -1)
                    cv2.putText(frame, emotion, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Ajustar o frame do vídeo para a resolução do label_video
            frame = cv2.resize(frame, (label_video.winfo_width(), label_video.winfo_height()))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            label_video.configure(image=frame)
            label_video.image = frame

            label_video.after(30, update_frame)
        else:
            cap.release()

    update_frame()

def select_source(label_video):
    source = filedialog.askopenfilename(parent=root, filetypes=[("Video files", ".mp4;.avi;*.mov")])
    if source:
        start_emotion_detection(source, label_video)

def select_webcam(label_video):
    start_emotion_detection(0, label_video)

root = tk.Tk()
root.title("Detecção de Emoções")
root.attributes('-fullscreen', True)  # Abrir em tela cheia

left_frame = tk.Frame(root, bg="#F0F0F0")
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root, bg="#F0F0F0")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

video_label = ttk.Label(right_frame, background="#000000")
video_label.pack(pady=10, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(right_frame, bg="#F0F0F0")
button_frame.pack(pady=10)

select_video_button = ttk.Button(button_frame, text="Selecionar Vídeo", command=lambda: select_source(video_label), style="Blue.TButton")
select_video_button.pack(side="left", padx=10)

select_webcam_button = ttk.Button(button_frame, text="Usar Webcam", command=lambda: select_webcam(video_label), style="Blue.TButton")
select_webcam_button.pack(side="left", padx=10)

quit_button = ttk.Button(right_frame, text="Sair", command=root.quit, style="Blue.TButton")
quit_button.pack(pady=10)

style = ttk.Style()
style.configure("Blue.TButton", padding=10, relief="flat", background="#0078D4", foreground="#000000", font=("Arial", 12), borderwidth=0)

update_graphs()
root.mainloop()
