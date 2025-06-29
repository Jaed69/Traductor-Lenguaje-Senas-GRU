# data_collector_modificado.py

import cv2
import mediapipe as mp
import numpy as np
import os
import shutil  # <-- IMPORTANTE: Añadimos esta librería para borrar carpetas

class SequenceDataCollector:
    def __init__(self, signs_to_collect, sequence_length=50, num_sequences=40):
        """
        Inicializa el recolector de datos.

        Args:
            signs_to_collect (list): Lista de strings con las señas a recolectar.
            sequence_length (int): Número de frames por secuencia de video.
            num_sequences (int): Número total de secuencias a recolectar por seña.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.all_signs = signs_to_collect
        self.sequence_length = sequence_length
        self.num_sequences_per_sign = num_sequences
        self.data_path = os.path.join('data', 'sequences')
        os.makedirs(self.data_path, exist_ok=True)

    def _extract_landmarks(self, hand_landmarks):
        """
        Extrae y normaliza los landmarks de una mano detectada en un frame.
        La normalización se hace restando las coordenadas del punto de la muñeca (landmark 0).
        """
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return landmarks

    def _get_collected_sequences_count(self, sign):
        """
        Verifica cuántas secuencias ya han sido recolectadas para una seña específica.
        """
        sign_path = os.path.join(self.data_path, sign)
        if not os.path.isdir(sign_path):
            return 0
        return len([name for name in os.listdir(sign_path) if name.endswith('.npy')])

    def _display_menu(self):
        """
        Muestra el menú dinámico al usuario con el estado de la recolección.
        """
        print("\n" + "="*50)
        print("     MENÚ DE RECOLECCIÓN DE DATOS DE LENGUAJE DE SEÑAS")
        print("="*50)
        print("Selecciona la seña para la cual deseas recolectar datos:")
        
        for i, sign in enumerate(self.all_signs):
            collected_count = self._get_collected_sequences_count(sign)
            print(f"   {i+1}. {sign:<10} | Recolectadas: {collected_count}/{self.num_sequences_per_sign}")
            
        print("\n   0. Salir")
        print("="*50)

    def _collect_for_sign(self, sign):
        """
        Función principal para recolectar secuencias para una única seña.
        """
        start_sequence = self._get_collected_sequences_count(sign)
        
        # --- INICIO DE LA MODIFICACIÓN ---
        if start_sequence >= self.num_sequences_per_sign:
            print(f"\n¡Ya has recolectado todas las secuencias para la seña '{sign}'!")
            
            # Bucle para asegurar una respuesta válida (s/n)
            while True:
                respuesta = input(f"¿Deseas BORRAR los datos existentes y volver a recolectar? (s/n): ").lower()
                if respuesta in ['s', 'n']:
                    break
                print("Respuesta no válida. Por favor, ingresa 's' para sí o 'n' para no.")

            if respuesta == 's':
                sign_path = os.path.join(self.data_path, sign)
                print(f"Borrando datos antiguos para '{sign}'...")
                if os.path.isdir(sign_path):
                    shutil.rmtree(sign_path)
                print("Datos borrados. Comenzando nueva recolección desde cero.")
                start_sequence = 0  # Reiniciar el contador de secuencias
            else:
                print("Operación cancelada. Volviendo al menú principal.")
                return # Vuelve al menú sin hacer nada
        # --- FIN DE LA MODIFICACIÓN ---
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            return

        # Bucle para recolectar las secuencias que faltan
        for seq_num in range(start_sequence, self.num_sequences_per_sign):
            # Mensaje de espera antes de empezar cada secuencia
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                frame = cv2.flip(frame, 1) # Voltear la imagen horizontalmente
                
                msg = f"Listo para '{sign}' | Secuencia {seq_num+1}. Presiona 'S' para empezar."
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Recolector de Datos', frame)
                
                if cv2.waitKey(10) & 0xFF == ord('s'):
                    break
            
            if not success: # Si la cámara se desconectó en el bucle de espera
                print("Error de cámara. Saliendo de la recolección.")
                break

            print(f"Recolectando secuencia {seq_num} para la seña '{sign}'...")
            
            sequence_data = []
            # Bucle para capturar los frames de una secuencia
            for frame_num in range(self.sequence_length):
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1) # Voltear la imagen horizontalmente
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                frame_landmarks = np.zeros(2 * 21 * 3)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    handedness_list = [h.classification[0].label for h in results.multi_handedness] if results.multi_handedness else []
                    
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        lm_data = self._extract_landmarks(hand_landmarks)
                        if len(handedness_list) > i and handedness_list[i] == 'Left':
                            frame_landmarks[0:63] = lm_data
                        elif len(handedness_list) > i and handedness_list[i] == 'Right':
                            frame_landmarks[63:126] = lm_data
                        else:
                            if i == 0: frame_landmarks[0:63] = lm_data
                            if i == 1: frame_landmarks[63:126] = lm_data

                sequence_data.append(frame_landmarks)

                cv2.putText(frame, f"Recolectando... Frame {frame_num+1}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Recolector de Datos', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Guardar la secuencia recolectada
            if len(sequence_data) == self.sequence_length:
                sequence_array = np.array(sequence_data)
                sign_path = os.path.join(self.data_path, sign)
                os.makedirs(sign_path, exist_ok=True)
                save_path = os.path.join(sign_path, str(seq_num))
                np.save(save_path, sequence_array)
                print(f"Secuencia {seq_num} para '{sign}' guardada.")

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """
        Ejecuta el bucle principal que muestra el menú y gestiona la recolección.
        """
        while True:
            self._display_menu()
            try:
                choice = int(input("Ingresa tu opción (número): "))
                if choice == 0:
                    print("Saliendo del programa. ¡Hasta luego!")
                    break
                elif 1 <= choice <= len(self.all_signs):
                    selected_sign = self.all_signs[choice - 1]
                    print(f"\nHas seleccionado recolectar para la seña: '{selected_sign}'")
                    self._collect_for_sign(selected_sign)
                else:
                    print("Opción no válida. Por favor, intenta de nuevo.")
            except ValueError:
                print("Entrada inválida. Por favor, ingresa un número.")
            except Exception as e:
                print(f"Ocurrió un error: {e}")


# --- EJECUCIÓN DEL PROGRAMA ---
if __name__ == '__main__':
    # Define la lista completa de señas que tu modelo reconocerá.
    signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
             'HOLA', 'GRACIAS', 'POR FAVOR']
             
    collector = SequenceDataCollector(signs_to_collect=signs, sequence_length=50, num_sequences=40)
    
    collector.run()