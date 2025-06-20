# real_time_translator_sequence.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os

class RealTimeSequenceTranslator:
    def __init__(self, model_path='data/sign_model_gru.h5', signs_path='data/signs.npy'):
        self.model = tf.keras.models.load_model(model_path)
        self.signs = np.load(signs_path)
        self.sequence_length = 30
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_threshold = 0.7 # Confianza mínima para mostrar predicción
        
        # ... (inicialización de MediaPipe y OpenCV como antes) ...
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

    def _extract_landmarks(self, hand_landmarks):
        # ... (sin cambios) ...
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return landmarks

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            current_landmarks = np.zeros(21 * 3) # Frame por defecto (sin mano)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                current_landmarks = self._extract_landmarks(hand_landmarks)

            self.sequence_buffer.append(current_landmarks)
            
            # Solo predecir si el búfer está lleno
            if len(self.sequence_buffer) == self.sequence_length:
                sequence_array = np.expand_dims(np.array(self.sequence_buffer), axis=0)
                
                prediction = self.model.predict(sequence_array)[0]
                predicted_idx = np.argmax(prediction)
                confidence = prediction[predicted_idx]

                if confidence > self.prediction_threshold:
                    predicted_sign = self.signs[predicted_idx]
                    cv2.putText(frame, f"{predicted_sign} ({confidence*100:.2f}%)",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            cv2.imshow('Traductor de Señas Dinámicas', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

# =================================================================
# !! AÑADE O VERIFICA QUE ESTE BLOQUE EXISTA AL FINAL DEL ARCHIVO !!
# =================================================================
if __name__ == '__main__':
    MODEL_PATH = 'data/sign_model_gru.h5'
    SIGNS_PATH = 'data/label_encoder.npy'

    # --- Verificación de archivos ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encontró el archivo del modelo en la ruta: {MODEL_PATH}")
        print("Asegúrate de haber ejecutado 'model_trainer_sequence.py' primero.")
    elif not os.path.exists(SIGNS_PATH):
        print(f"ERROR: No se encontró el archivo de etiquetas en la ruta: {SIGNS_PATH}")
        print("Asegúrate de que el archivo 'label_encoder.npy' exista en la carpeta 'data'.")
    else:
        print("Modelo y etiquetas encontrados. Iniciando traductor...")
        translator = RealTimeSequenceTranslator(model_path=MODEL_PATH, signs_path=SIGNS_PATH)
        translator.run()
