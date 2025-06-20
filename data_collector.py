# data_collector_sequence.py (versión modificada)

import cv2
import mediapipe as mp
import numpy as np
import os

class SequenceDataCollector:
    def __init__(self, signs_to_collect=['J', 'Z'], sequence_length=30, num_sequences=30):
        # ... (inicialización de MediaPipe igual que antes)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.signs_to_collect = signs_to_collect
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.data_path = os.path.join('data', 'sequences')
        os.makedirs(self.data_path, exist_ok=True)

    def _extract_landmarks(self, hand_landmarks):
        # ... (sin cambios, extrae y normaliza los landmarks de un frame)
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return landmarks

    def collect_sequences(self):
        cap = cv2.VideoCapture(0)
        
        for sign in self.signs_to_collect:
            for seq_num in range(self.num_sequences):
                # Mensaje para iniciar la captura de la secuencia
                while True:
                    success, frame = cap.read()
                    cv2.putText(frame, f"Listo para '{sign}' Seq_Num {seq_num}. Presiona 'S' para empezar.",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Data Collector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break
                
                print(f"Recolectando secuencia {seq_num} para la seña '{sign}'...")
                
                sequence_data = []
                for frame_num in range(self.sequence_length):
                    success, frame = cap.read()
                    if not success: continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
                        landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                        sequence_data.append(landmarks)
                    else:
                        # Si no se detecta la mano, añadimos un frame de ceros
                        sequence_data.append(np.zeros(21 * 3))

                    cv2.putText(frame, f"Recolectando... Frame {frame_num}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Data Collector', frame)
                    cv2.waitKey(1) # Pequeña pausa

                # Guardar la secuencia
                sequence_array = np.array(sequence_data)
                save_path = os.path.join(self.data_path, sign, str(seq_num))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, sequence_array)

        cap.release()
        cv2.destroyAllWindows()

# Al ejecutar, puedes instanciar la clase así:
if __name__ == '__main__':
    # Define las señas dinámicas que quieres aprender. Empieza con pocas.
    # Recolecta unas 30-50 secuencias por cada seña para empezar.
    collector = SequenceDataCollector(signs_to_collect=['J', 'Z', 'HOLA'], sequence_length=30, num_sequences=40)
    collector.collect_sequences()