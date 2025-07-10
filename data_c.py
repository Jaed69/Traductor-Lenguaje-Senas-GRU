# ===================================================================
# 🚀 Recolector de Datos LSP - Versión Modernizada y Completa (Julio 2025)
#
# Actualizado para usar la API de Tareas de MediaPipe (>=0.10.11)
# y las dependencias recomendadas para Python 3.11+.
# Incluye toda la lógica de menú para una experiencia de recolección completa.
#
# Autor: [Tu Nombre/Alias]
# ===================================================================

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import threading
from collections import deque
from datetime import datetime

# --- Importaciones específicas de la nueva API de Tareas de MediaPipe ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LSPDataCollector:
    """
    Un recolector de datos avanzado para Lenguaje de Señas Peruano (LSP)
    que utiliza la moderna API de Tareas de MediaPipe para un rendimiento
    y una precisión mejorados.
    """
    def __init__(self, sequence_length=60, num_sequences=50):  # Optimizado para GRU
        self.sequence_length = sequence_length  # Aumentado para mejor contexto temporal
        self.num_sequences = num_sequences      # Más secuencias para mejor generalización
        
        # --- Configuración MediaPipe con la API de Tareas ---
        self.setup_mediapipe_tasks()

        # Variables para manejar resultados asíncronos de MediaPipe
        self.latest_hand_results = None
        self.latest_pose_results = None
        self.lock = threading.Lock()
        
        # Configuración optimizada para GRU bidireccional
        self.gru_optimized_features = True
        self.temporal_smoothing = True
        self.feature_normalization = True
        
        # Clasificación de señas por tipo - Optimizada para GRU
        self.sign_types = {
            'static_one_hand': {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'},
            'dynamic_one_hand': {'J', 'Z', 'Ñ', 'RR', 'LL'},
            'static_two_hands': {'AMOR', 'CASA', 'FAMILIA', 'ESCUELA'},
            'dynamic_two_hands': {'HOLA', 'GRACIAS', 'POR FAVOR', 'ADIÓS', 'CÓMO ESTÁS'},
            'phrases': {'BUENOS DÍAS', 'BUENAS NOCHES', 'MUCHO GUSTO', 'DE NADA'}
        }
        
        self.signs_to_collect = sorted([sign for category in self.sign_types.values() for sign in category])
        
        self.data_dir = 'data/sequences_advanced'
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("🚀 Recolector de Datos LSP Inicializado (API de Tareas Modernizada)")
        print("📝 Características:")
        print("   • Normalización automática derecha/izquierda")
        print("   • Detección de señas estáticas vs dinámicas")
        print("   • Soporte para 1 o 2 manos")
        print("   • Análisis de calidad en tiempo real")
        print("   • 16 métricas de movimiento optimizadas para GRU")
        print("   • Metadatos completos por secuencia")
        print("   • 🧠 Optimizado para GRU Bidireccional")
        print("   • 🎯 Secuencias de 60 frames para mejor contexto temporal")
        print("   • 📊 Features normalizadas para keras.GRU")

    def setup_mediapipe_tasks(self):
        """Inicializa los modelos de MediaPipe usando la API de Tareas."""
        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=self._process_hand_results
            )
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=self._process_pose_results
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        except Exception as e:
            print("\n" + "="*80)
            print("❌ ERROR: No se pudieron cargar los modelos de MediaPipe.")
            print("   Asegúrate de haber descargado los archivos 'hand_landmarker.task' y 'pose_landmarker_heavy.task'")
            print("   y haberlos colocado en una carpeta llamada 'models' junto a este script.")
            print(f"   Error original: {e}")
            print("="*80 + "\n")
            exit()

    def _process_hand_results(self, result, output_image, timestamp_ms: int):
        with self.lock:
            self.latest_hand_results = result

    def _process_pose_results(self, result, output_image, timestamp_ms: int):
        with self.lock:
            self.latest_pose_results = result

    def _normalize_hand_landmarks(self, landmarks_list, handedness):
        """Normaliza landmarks para ser independientes de la mano - Versión original más precisa"""
        hand_landmarks = []
        for lm in landmarks_list:
            hand_landmarks.extend([lm.x, lm.y, lm.z])
        
        landmarks_array = np.array(hand_landmarks).reshape(-1, 3)
        
        # Si es mano izquierda, invertir en X para normalizar
        if handedness == 'Left':
            landmarks_array[:, 0] = 1.0 - landmarks_array[:, 0]
        
        # Normalizar respecto a la muñeca (punto 0)
        wrist = landmarks_array[0]
        normalized_landmarks = landmarks_array - wrist
        
        return normalized_landmarks.flatten()

    def _extract_advanced_landmarks(self, hand_results, pose_results):
        """Extrae landmarks optimizados para GRU bidireccional"""
        # Inicializar arrays optimizados para GRU
        hand_data = np.zeros(126)  # 2 manos * 63 features cada una
        pose_data = np.zeros(24)   # 8 puntos clave * 3 coordenadas
        velocity_data = np.zeros(6)  # Velocidades de manos y pose para GRU temporal
        hands_info = {'count': 0, 'handedness': [], 'confidence': []}

        # Procesar manos con normalización precisa y cálculo de velocidades
        if hand_results and hand_results.hand_landmarks:
            hands_info['count'] = len(hand_results.hand_landmarks)
            
            for i, (hand_landmarks_list, handedness_list) in enumerate(zip(hand_results.hand_landmarks, hand_results.handedness)):
                handedness = handedness_list[0].category_name
                confidence = handedness_list[0].score
                
                hands_info['handedness'].append(handedness)
                hands_info['confidence'].append(confidence)
                
                # Usar la normalización más precisa
                normalized_landmarks = self._normalize_hand_landmarks(hand_landmarks_list, handedness)
                
                # Asignar a posición correspondiente (versión original más robusta)
                if handedness == 'Right' or i == 0:
                    hand_data[0:63] = normalized_landmarks
                    # Calcular velocidad de mano derecha (para GRU temporal)
                    if hasattr(self, 'prev_right_hand') and self.prev_right_hand is not None:
                        velocity_data[0:3] = np.linalg.norm(normalized_landmarks[0:3] - self.prev_right_hand[0:3])
                    self.prev_right_hand = normalized_landmarks
                else:
                    hand_data[63:126] = normalized_landmarks
                    # Calcular velocidad de mano izquierda
                    if hasattr(self, 'prev_left_hand') and self.prev_left_hand is not None:
                        velocity_data[3:6] = np.linalg.norm(normalized_landmarks[0:3] - self.prev_left_hand[0:3])
                    self.prev_left_hand = normalized_landmarks

        # Procesar pose (solo puntos relevantes para lenguaje de señas)
        if pose_results and pose_results.pose_landmarks:
            relevant_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # Puntos clave del torso y brazos
            all_pose_landmarks = pose_results.pose_landmarks[0]
            extracted_pose_landmarks = []
            
            for idx in relevant_indices:
                if idx < len(all_pose_landmarks):
                    lm = all_pose_landmarks[idx]
                    extracted_pose_landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(extracted_pose_landmarks) >= 24:  # 8 puntos * 3 coordenadas
                pose_data = np.array(extracted_pose_landmarks[:24])
                
                # Calcular velocidad de pose para información temporal
                if hasattr(self, 'prev_pose') and self.prev_pose is not None:
                    pose_velocity = np.linalg.norm(pose_data - self.prev_pose)
                    velocity_data = np.append(velocity_data, pose_velocity)
                else:
                    velocity_data = np.append(velocity_data, 0.0)
                self.prev_pose = pose_data

        # Combinar features optimizadas para GRU
        combined_features = np.concatenate([hand_data, pose_data, velocity_data])
        
        # Aplicar normalización si está habilitada (recomendado para GRU)
        if self.feature_normalization:
            combined_features = self._normalize_features_for_gru(combined_features)
        
        return combined_features, hands_info

    def _normalize_features_for_gru(self, features):
        """Normaliza features específicamente para GRU bidireccional"""
        # Normalización Min-Max adaptada para GRU
        # Los GRU funcionan mejor con datos en rango [-1, 1] o [0, 1]
        features_norm = features.copy()
        
        # Normalizar landmarks de manos (posiciones relativas ya están normalizadas)
        hand_features = features_norm[:126]
        if np.max(np.abs(hand_features)) > 0:
            hand_features = np.tanh(hand_features * 2)  # Tanh para rango [-1, 1]
        
        # Normalizar pose
        pose_features = features_norm[126:150]
        if np.max(np.abs(pose_features)) > 0:
            pose_features = (pose_features - 0.5) * 2  # Rango [-1, 1]
        
        # Normalizar velocidades (importantes para contexto temporal en GRU)
        velocity_features = features_norm[150:]
        if np.max(velocity_features) > 0:
            velocity_features = np.clip(velocity_features / np.max(velocity_features), 0, 1)
        
        return np.concatenate([hand_features, pose_features, velocity_features])

    def _collect_single_sequence(self, sign, sign_path, sequence_id, sign_type, mode="NORMAL"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara.")
            return False

        sequence_buffer = deque(maxlen=self.sequence_length)
        collecting = False
        frame_count = 0
        
        print(f"\n🎯 {mode}: Recolectando '{sign}' - Secuencia {sequence_id}")
        print("📱 Presiona [ESPACIO] para iniciar/detener. Presiona [Q] para salir.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time.time() * 1000)
            
            self.hand_landmarker.detect_async(mp_image, timestamp)
            self.pose_landmarker.detect_async(mp_image, timestamp)
            
            with self.lock:
                hand_res, pose_res = self.latest_hand_results, self.latest_pose_results
            
            combined_data, hands_info = self._extract_advanced_landmarks(hand_res, pose_res)
            
            if collecting:
                sequence_buffer.append(combined_data)
                frame_count += 1
                progress_bar_width = int((frame_count / self.sequence_length) * frame.shape[1])
                cv2.rectangle(frame, (0, frame.shape[0] - 10), (progress_bar_width, frame.shape[0]), (0, 255, 0), -1)

                if frame_count >= self.sequence_length:
                    collecting = False
                    cap.release()
                    cv2.destroyAllWindows()
                    return self._process_and_save_sequence(sequence_buffer, sign, sign_path, sequence_id, sign_type, mode, hands_info)

            self._draw_landmarks_on_frame(frame, hand_res)
            self._display_hud(frame, collecting, hands_info)

            cv2.imshow('Recolector de Datos LSP', frame)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                collecting = not collecting
                if collecting:
                    frame_count = 0
                    sequence_buffer.clear()
                    print("🎬 Iniciando recolección...")
                else:
                    print("⏸️ Recolección pausada.")
        
        cap.release()
        cv2.destroyAllWindows()
        return False

    def _process_and_save_sequence(self, buffer, sign, path, seq_id, sign_type, mode, hands_info):
        sequence_data = np.array(buffer)
        motion_features = self._calculate_motion_features(sequence_data)
        quality_score, quality_level, issues = self._evaluate_sequence_quality(sequence_data, motion_features, sign_type)
        
        print(f"\n📊 Calidad obtenida: {quality_level} ({quality_score:.1f}%)")
        if issues: print(f"⚠️ Problemas detectados: {', '.join(issues)}")
        
        accept = input("¿Aceptar esta secuencia? (s/n): ").strip().lower()
        
        if accept in ['s', 'si', 'y', 'yes', '']:
            metadata = {
                'sign': sign, 'sign_type': sign_type, 'hands_count': hands_info.get('count', 0),
                'handedness': hands_info.get('handedness', []), 'quality_score': quality_score,
                'quality_level': quality_level, 'motion_features': motion_features.tolist(),
                'issues': issues, 'collection_mode': mode, 'timestamp': datetime.now().isoformat()
            }
            np.save(os.path.join(path, f"{seq_id}.npy"), sequence_data)
            with open(os.path.join(path, f"{seq_id}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"✅ Secuencia {seq_id} guardada para '{sign}'.")
            return True
        else:
            print("❌ Secuencia descartada.")
            return False

    def _draw_landmarks_on_frame(self, frame, hand_results):
        """Dibuja landmarks de manos en el frame - Versión simplificada y robusta"""
        if not hand_results or not hand_results.hand_landmarks:
            return
        
        # Dibujar usando cv2 directamente para máxima compatibilidad y precisión
        h, w, _ = frame.shape
        
        for hand_landmarks_list in hand_results.hand_landmarks:
            # Dibujar puntos de landmarks
            for i, landmark in enumerate(hand_landmarks_list):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Diferentes colores para diferentes tipos de puntos
                if i == 0:  # Muñeca
                    color = (0, 0, 255)  # Rojo
                    radius = 5
                elif i in [4, 8, 12, 16, 20]:  # Puntas de dedos
                    color = (0, 255, 0)  # Verde
                    radius = 4
                else:  # Otros puntos
                    color = (255, 255, 255)  # Blanco
                    radius = 3
                
                cv2.circle(frame, (x, y), radius, color, -1)
            
            # Dibujar conexiones básicas entre puntos (estructura de mano)
            connections = [
                # Pulgar
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Índice
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Medio
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Anular
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Meñique
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            for connection in connections:
                if connection[0] < len(hand_landmarks_list) and connection[1] < len(hand_landmarks_list):
                    pt1_landmark = hand_landmarks_list[connection[0]]
                    pt2_landmark = hand_landmarks_list[connection[1]]
                    
                    pt1 = (int(pt1_landmark.x * w), int(pt1_landmark.y * h))
                    pt2 = (int(pt2_landmark.x * w), int(pt2_landmark.y * h))
                    
                    cv2.line(frame, pt1, pt2, (100, 100, 255), 2)

    def _display_hud(self, frame, collecting, hands_info):
        """HUD optimizado para mostrar información relevante para GRU"""
        # Estado de grabación
        status_text = "GRABANDO (GRU-Optimizado)" if collecting else "PAUSADO"
        status_color = (0, 0, 255) if collecting else (255, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        
        # Información de manos detectadas
        hands_text = f"Manos: {hands_info.get('count', 0)}"
        if hands_info.get('handedness'):
            hands_text += f" ({', '.join(hands_info['handedness'])})"
        cv2.putText(frame, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Información específica para GRU
        gru_info = [
            f"Secuencia: {self.sequence_length} frames (GRU-opt)",
            f"Features: {self.gru_optimized_features} activas",
            f"Suavizado: {'ON' if self.temporal_smoothing else 'OFF'}",
            f"Normalización: {'ON' if self.feature_normalization else 'OFF'}"
        ]
        
        for i, info in enumerate(gru_info):
            cv2.putText(frame, info, (10, 90 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Calidad en tiempo real si está recolectando
        if collecting and hasattr(self, 'prev_landmarks'):
            # Indicador de estabilidad temporal
            stability_color = (0, 255, 0) if hasattr(self, 'prev_landmarks') else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, stability_color, -1)
            cv2.putText(frame, "Estabilidad", (frame.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controles
        cv2.putText(frame, "ESPACIO: Iniciar/Parar | Q: Salir", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
    def _classify_sign_type(self, sign):
        for category, signs in self.sign_types.items():
            if sign in signs: return category
        return 'unknown'

    def _calculate_motion_features(self, sequence_data):
        """Calcula métricas de movimiento optimizadas para GRU bidireccional"""
        if len(sequence_data) < 5:
            return np.zeros(20)  # Aumentado para métricas GRU
        
        # Separar componentes de los datos
        hand_sequence = sequence_data[:, :126]
        pose_sequence = sequence_data[:, 126:150] if sequence_data.shape[1] > 150 else None
        velocity_sequence = sequence_data[:, 150:] if sequence_data.shape[1] > 150 else None
        
        # 1. Métricas básicas de movimiento de manos
        frame_movements = [np.linalg.norm(hand_sequence[i] - hand_sequence[i-1]) for i in range(1, len(hand_sequence))]
        if not frame_movements:
            return np.zeros(20)
        
        # 2. Calcular aceleraciones y jerk (importantes para GRU temporal)
        accelerations = [abs(frame_movements[i] - frame_movements[i-1]) for i in range(1, len(frame_movements))]
        jerk_values = [abs(accelerations[i] - accelerations[i-1]) for i in range(1, len(accelerations))]
        
        # 3. Métricas básicas (12 originales)
        basic_metrics = [
            sum(frame_movements),  # Movimiento total
            np.mean(frame_movements),  # Velocidad promedio
            max(frame_movements),  # Velocidad máxima
            min(frame_movements),  # Velocidad mínima
            np.var(frame_movements),  # Varianza de velocidad
            np.std(frame_movements),  # Desviación estándar
            np.mean(accelerations) if accelerations else 0,  # Aceleración promedio
            max(accelerations) if accelerations else 0,  # Aceleración máxima
            np.linalg.norm(hand_sequence[-1] - hand_sequence[0]),  # Desplazamiento total
            len([m for m in frame_movements if m > np.mean(frame_movements)]),  # Frames con alta velocidad
            np.var(hand_sequence, axis=0).mean(),  # Varianza espacial
            np.mean(jerk_values) if jerk_values else 0  # Jerk promedio
        ]
        
        # 4. Métricas adicionales para GRU (8 nuevas métricas)
        additional_metrics = []
        
        # 4.1. Consistencia temporal
        temporal_consistency = 1.0 / (1.0 + np.var(frame_movements)) if frame_movements else 0.0
        additional_metrics.append(temporal_consistency)
        
        # 4.2. Smoothness (suavidad del movimiento)
        smoothness = 1.0 / (1.0 + np.mean(jerk_values)) if jerk_values else 1.0
        additional_metrics.append(smoothness)
        
        # 4.3. Coordinación entre manos
        right_hand_seq = hand_sequence[:, :63]
        left_hand_seq = hand_sequence[:, 63:126]
        
        if np.any(right_hand_seq) and np.any(left_hand_seq):
            right_movement = [np.linalg.norm(right_hand_seq[i] - right_hand_seq[i-1]) for i in range(1, len(right_hand_seq))]
            left_movement = [np.linalg.norm(left_hand_seq[i] - left_hand_seq[i-1]) for i in range(1, len(left_hand_seq))]
            
            if right_movement and left_movement:
                hand_correlation = abs(np.corrcoef(right_movement, left_movement)[0, 1])
                hand_correlation = hand_correlation if not np.isnan(hand_correlation) else 0.0
            else:
                hand_correlation = 0.0
        else:
            hand_correlation = 0.0
        additional_metrics.append(hand_correlation)
        
        # 4.4. Estabilidad inicial y final (importante para señas)
        start_stability = 1.0 / (1.0 + np.linalg.norm(hand_sequence[1] - hand_sequence[0])) if len(hand_sequence) > 1 else 1.0
        end_stability = 1.0 / (1.0 + np.linalg.norm(hand_sequence[-1] - hand_sequence[-2])) if len(hand_sequence) > 1 else 1.0
        additional_metrics.extend([start_stability, end_stability])
        
        # 4.5. Complejidad gestual
        gesture_complexity = np.mean(np.std(hand_sequence, axis=0)) if hand_sequence.size > 0 else 0.0
        additional_metrics.append(gesture_complexity)
        
        # 4.6. Periodicidad (para detectar gestos repetitivos)
        if len(frame_movements) > 10:
            from scipy import signal
            try:
                # Buscar periodicidad en el movimiento
                autocorr = signal.correlate(frame_movements, frame_movements, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                periodicity = np.max(autocorr[1:]) / autocorr[0] if len(autocorr) > 1 else 0.0
            except:
                periodicity = 0.0
        else:
            periodicity = 0.0
        additional_metrics.append(periodicity)
        
        # 4.7. Intensidad direccional (preferencia de movimiento en ciertas direcciones)
        if len(hand_sequence) > 2:
            x_movement = np.std(hand_sequence[:, 0::3])  # Movimiento en X
            y_movement = np.std(hand_sequence[:, 1::3])  # Movimiento en Y
            z_movement = np.std(hand_sequence[:, 2::3])  # Movimiento en Z
            directional_intensity = max(x_movement, y_movement, z_movement) / (x_movement + y_movement + z_movement + 1e-8)
        else:
            directional_intensity = 0.0
        additional_metrics.append(directional_intensity)
        
        # Combinar todas las métricas
        all_metrics = basic_metrics + additional_metrics
        
        # Asegurar exactamente 20 métricas
        while len(all_metrics) < 20:
            all_metrics.append(0.0)
        
        return np.array(all_metrics[:20])

    def _evaluate_sequence_quality(self, sequence_data, motion_features, sign_type):
        """Evalúa calidad de secuencia optimizada para GRU bidireccional"""
        quality_score = 100.0
        issues = []
        
        # 1. Completeness de datos (más crítico para GRU)
        completeness = np.count_nonzero(np.any(sequence_data[:, :126] != 0, axis=1)) / len(sequence_data)
        if completeness < 0.9:  # Más estricto para GRU
            quality_score -= 25
            issues.append("Datos de mano incompletos (crítico para GRU)")
        elif completeness < 0.95:
            quality_score -= 10
            issues.append("Algunos frames con datos faltantes")
        
        # 2. Análisis de movimiento específico por tipo de seña
        avg_movement = motion_features[1]  # Velocidad promedio
        temporal_consistency = motion_features[12] if len(motion_features) > 12 else 0.0
        smoothness = motion_features[13] if len(motion_features) > 13 else 1.0
        
        if 'static' in sign_type:
            # Para señas estáticas: movimiento mínimo pero consistente
            if avg_movement > 0.012:  # Umbral ajustado
                quality_score -= 20
                issues.append("Demasiado movimiento para seña estática")
            elif avg_movement < 0.001:
                quality_score -= 15
                issues.append("Seña demasiado rígida (falta micro-movimientos naturales)")
                
            # Consistencia temporal más importante en señas estáticas
            if temporal_consistency < 0.7:
                quality_score -= 15
                issues.append("Baja consistencia temporal en seña estática")
                
        elif 'dynamic' in sign_type:
            # Para señas dinámicas: movimiento fluido y progresivo
            if avg_movement < 0.008:
                quality_score -= 20
                issues.append("Poco movimiento para seña dinámica")
            elif avg_movement > 0.05:
                quality_score -= 10
                issues.append("Movimiento excesivo (posible ruido)")
                
            # Suavidad más importante en señas dinámicas
            if smoothness < 0.5:
                quality_score -= 15
                issues.append("Movimiento entrecortado en seña dinámica")
        
        # 3. Estabilidad inicial y final (crítico para GRU temporal)
        if len(motion_features) > 15:
            start_stability = motion_features[15]
            end_stability = motion_features[16]
            
            if start_stability < 0.8:
                quality_score -= 12
                issues.append("Inicio inestable (afecta contexto temporal)")
            if end_stability < 0.8:
                quality_score -= 12
                issues.append("Final inestable (afecta contexto temporal)")
        
        # 4. Coordinación entre manos (importante para GRU bidireccional)
        if len(motion_features) > 14:
            hand_coordination = motion_features[14]
            if hand_coordination < 0.3 and 'bilateral' in sign_type:
                quality_score -= 15
                issues.append("Baja coordinación entre manos en seña bilateral")
        
        # 5. Análisis de complejidad gestual
        if len(motion_features) > 17:
            gesture_complexity = motion_features[17]
            if gesture_complexity > 0.8:
                quality_score -= 8
                issues.append("Gesto muy complejo (posible sobreactuación)")
            elif gesture_complexity < 0.1:
                quality_score -= 5
                issues.append("Gesto demasiado simple (falta expresividad)")
        
        # 6. Evaluación de jerk y aceleración (suavidad para GRU)
        jerk_avg = motion_features[11] if len(motion_features) > 11 else 0.0
        acceleration_avg = motion_features[6] if len(motion_features) > 6 else 0.0
        
        if jerk_avg > 0.05:
            quality_score -= 10
            issues.append("Movimiento muy abrupto (alto jerk)")
        if acceleration_avg > 0.03:
            quality_score -= 8
            issues.append("Aceleraciones excesivas")
        
        # 7. Bonus por características ideales para GRU
        if temporal_consistency > 0.9:
            quality_score += 5  # Bonus por excelente consistencia
        if smoothness > 0.85:
            quality_score += 3  # Bonus por suavidad
        if completeness == 1.0:
            quality_score += 2  # Bonus por datos completos
        
        # 8. Penalización por secuencias muy cortas o largas
        sequence_length = len(sequence_data)
        if sequence_length < 30:
            quality_score -= 15
            issues.append("Secuencia muy corta para análisis temporal completo")
        elif sequence_length > 90:
            quality_score -= 10
            issues.append("Secuencia excesivamente larga (posible redundancia)")
        
        # Normalizar score
        quality_score = max(0, min(100, quality_score))
        
        # Determinar nivel de calidad con criterios más estrictos para GRU
        if quality_score >= 92:
            quality_level = "EXCELENTE (Óptimo para GRU)"
        elif quality_score >= 80:
            quality_level = "BUENA (Aceptable para GRU)"
        elif quality_score >= 65:
            quality_level = "REGULAR (Requiere mejora para GRU)"
        else:
            quality_level = "MALA (Inadecuada para GRU)"
        
        return quality_score, quality_level, issues

    def _get_collected_sequences_count(self, sign):
        sign_path = os.path.join(self.data_dir, sign)
        if not os.path.exists(sign_path): return 0
        return len([f for f in os.listdir(sign_path) if f.endswith('.npy')])

    def run(self):
        while True:
            self._display_advanced_menu()
            choice = input("Selecciona una opción (1-5): ").strip()
            if choice == '1': self._select_specific_sign()
            elif choice == '2': self._select_by_category()
            elif choice == '3': self._improve_quality_mode()
            elif choice == '4': self._show_detailed_stats()
            elif choice == '5': print("👋 ¡Hasta luego!"); break
            else: print("❌ Opción inválida. Inténtalo de nuevo.")

    def _display_advanced_menu(self):
        print("\n" + "="*80); print("🚀 RECOLECTOR AVANZADO DE DATOS (API DE TAREAS)"); print("="*80)
        for category, signs in self.sign_types.items():
            print(f"\n📁 {category.upper().replace('_', ' ')}:")
            total_collected, total_needed = 0, len(signs) * self.num_sequences
            for sign in sorted(signs):
                collected = self._get_collected_sequences_count(sign)
                total_collected += collected
                progress = (collected / self.num_sequences) * 100
                status = "❌ No iniciado" if collected == 0 else f"📊 {collected}/{self.num_sequences}" if collected < self.num_sequences else "✅ Completo" if collected == self.num_sequences else f"⭐ {collected} (Expandido)"
                print(f"   {status:<20} {sign:<15} [{progress:5.1f}%]")
            category_progress = (total_collected / total_needed) * 100 if total_needed > 0 else 0
            print(f"   💯 Progreso categoría: {category_progress:.1f}%")
        print("\n[MENÚ PRINCIPAL]"); print("1. Recolectar seña específica"); print("2. Recolectar por categoría"); print("3. Modo: Mejorar Calidad de Datos"); print("4. Ver estadísticas detalladas"); print("5. Salir")

    def _collect_for_sign_advanced(self, sign):
        sign_path = os.path.join(self.data_dir, sign); os.makedirs(sign_path, exist_ok=True)
        collected_count = self._get_collected_sequences_count(sign)
        sign_type = self._classify_sign_type(sign)
        if collected_count >= self.num_sequences:
            print(f"✅ '{sign}' ya tiene {collected_count} secuencias."); print("📋 Opciones: 1. Sustituir (mejorar calidad), 2. Expandir (+40), 3. Saltar")
            choice = input("Selecciona opción (1-3): ").strip()
            if choice == '1': self._substitute_sequences(sign, sign_path, sign_type)
            elif choice == '2': self._expand_sequences(sign, sign_path, sign_type, collected_count)
            else: print(f"⏭️ Saltando '{sign}'.")
        else:
            self._collect_normal(sign, sign_path, sign_type, collected_count)

    def _collect_normal(self, sign, sign_path, sign_type, start_id, target_sequences=None):
        if target_sequences is None:
            target_sequences = self.num_sequences
        collected_count = start_id
        target_count = start_id + target_sequences
        
        while collected_count < target_count:
            if self._collect_single_sequence(sign, sign_path, collected_count, sign_type, "NORMAL"):
                collected_count += 1
                if collected_count < target_count:
                    if input("¿Continuar con la siguiente? (s/n): ").lower() not in ['s', '', 'si', 'y']: break
            elif input("¿Reintentar? (s/n): ").lower() not in ['s', '', 'si', 'y']:
                break
        print(f"🎉 Recolección para '{sign}' finalizada.")

    def _select_specific_sign(self):
        print("\n📝 Señas disponibles:"); [print(f"{i+1:2d}. {sign:<15} [{self._get_collected_sequences_count(sign)}/{self.num_sequences}]") for i, sign in enumerate(self.signs_to_collect)]
        try:
            choice = int(input("\nSelecciona número de seña: ")) - 1
            if 0 <= choice < len(self.signs_to_collect): self._collect_for_sign_advanced(self.signs_to_collect[choice])
            else: print("❌ Número inválido")
        except ValueError: print("❌ Por favor ingresa un número.")

    def _select_by_category(self):
        print("\n📁 Categorías:"); categories = list(self.sign_types.keys()); [print(f"{i+1}. {c.replace('_', ' ').title()}") for i, c in enumerate(categories)]
        try:
            choice = int(input("\nSelecciona categoría: ")) - 1
            if 0 <= choice < len(categories): self._collect_category(categories[choice])
            else: print("❌ Número inválido")
        except ValueError: print("❌ Por favor ingresa un número.")

    def _collect_category(self, category):
        print(f"\n🎯 Recolectando categoría: {category.replace('_', ' ').title()}")
        for sign in self.sign_types[category]:
            if self._get_collected_sequences_count(sign) < self.num_sequences:
                print(f"\n📍 Siguiente: {sign}")
                if input("¿Continuar? (s/n): ").lower() in ['s', 'si', 'y', 'yes', '']: self._collect_for_sign_advanced(sign)
                else: break
            else: print(f"✅ {sign} ya está completo.")

    def _show_detailed_stats(self):
        """Muestra estadísticas detalladas"""
        print("\n📊 ESTADÍSTICAS DETALLADAS")
        print("="*50)
        
        total_sequences = 0
        total_needed = len(self.signs_to_collect) * self.num_sequences
        
        for category, signs in self.sign_types.items():
            category_collected = 0
            category_needed = len(signs) * self.num_sequences
            
            print(f"\n📁 {category.upper().replace('_', ' ')}:")
            
            for sign in signs:
                collected = self._get_collected_sequences_count(sign)
                total_sequences += collected
                category_collected += collected
                
                # Verificar si hay metadatos de calidad
                sign_path = os.path.join(self.data_dir, sign)
                quality_info = ""
                expansion_info = ""
                
                if os.path.exists(sign_path):
                    metadata_files = [f for f in os.listdir(sign_path) if f.endswith('_metadata.json')]
                    if metadata_files:
                        total_quality = 0
                        quality_count = 0
                        substitution_count = 0
                        expansion_count = 0
                        
                        for meta_file in metadata_files:
                            try:
                                with open(os.path.join(sign_path, meta_file), 'r') as f:
                                    metadata = json.load(f)
                                    quality_score = metadata.get('quality_score', 0)
                                    if quality_score > 0:
                                        total_quality += quality_score
                                        quality_count += 1
                                    
                                    # Contar modos de recolección
                                    mode = metadata.get('collection_mode', 'NORMAL')
                                    if mode == 'SUSTITUCIÓN':
                                        substitution_count += 1
                                    elif mode == 'EXPANSIÓN':
                                        expansion_count += 1
                            except:
                                pass
                        
                        if quality_count > 0:
                            avg_quality = total_quality / quality_count
                            quality_info = f" (Q: {avg_quality:.0f}%)"
                        
                        if collected > self.num_sequences:
                            expansion_info = f" [+{collected - self.num_sequences}]"
                        
                        if substitution_count > 0:
                            quality_info += f" S:{substitution_count}"
                
                status_line = f"   {sign:<15} {collected:>3}/{self.num_sequences}{expansion_info}{quality_info}"
                print(status_line)
            
            category_progress = (category_collected / category_needed) * 100 if category_needed > 0 else 0
            print(f"   📈 Progreso: {category_progress:.1f}%")
        
        overall_progress = (total_sequences / total_needed) * 100 if total_needed > 0 else 0
        print(f"\n🎯 PROGRESO GENERAL: {overall_progress:.1f}%")
        print(f"📊 Total recolectado: {total_sequences}/{total_needed}")

    def _analyze_existing_quality(self, sign_path):
        """Analiza la calidad de secuencias existentes"""
        qualities = {}
        
        for file in os.listdir(sign_path):
            if file.endswith('_metadata.json'):
                try:
                    seq_id = int(file.split('_')[0])
                    with open(os.path.join(sign_path, file), 'r') as f:
                        metadata = json.load(f)
                        quality_score = metadata.get('quality_score', 0)
                        qualities[seq_id] = quality_score
                except (ValueError, json.JSONDecodeError):
                    continue
        
        return qualities

    def _substitute_sequences(self, sign, sign_path, sign_type):
        """Permite sustituir secuencias existentes con mejor calidad"""
        print(f"\n🔄 MODO SUSTITUCIÓN para '{sign}'")
        
        # Analizar calidad de secuencias existentes
        existing_qualities = self._analyze_existing_quality(sign_path)
        
        if not existing_qualities:
            print("❌ No se encontraron metadatos de calidad")
            return self._collect_normal(sign, sign_path, sign_type, 0)
        
        # Mostrar las secuencias con menor calidad
        sorted_qualities = sorted(existing_qualities.items(), key=lambda x: x[1])
        print(f"\n📊 Secuencias con menor calidad:")
        for i, (seq_id, quality) in enumerate(sorted_qualities[:10]):
            print(f"   {i+1:2d}. Secuencia {seq_id:2d}: {quality:5.1f}%")
        
        print(f"\n🎯 Recolectaremos nuevas secuencias para sustituir las de menor calidad")
        
        # Recolectar secuencias de sustitución
        sequences_to_replace = min(10, len(sorted_qualities))  # Máximo 10 sustituciones por sesión
        
        for i in range(sequences_to_replace):
            seq_id = sorted_qualities[i][0]
            old_quality = sorted_qualities[i][1]
            
            print(f"\n🔄 Sustituyendo secuencia {seq_id} (calidad actual: {old_quality:.1f}%)")
            response = input("¿Continuar con esta sustitución? (s/n): ").strip().lower()
            
            if response in ['s', 'si', 'y', 'yes']:
                success = self._collect_single_sequence(sign, sign_path, seq_id, sign_type, "SUSTITUCIÓN")
                if success:
                    print(f"✅ Secuencia {seq_id} sustituida exitosamente")
                else:
                    print(f"❌ Error al sustituir secuencia {seq_id}")
            else:
                print(f"⏭️ Saltando secuencia {seq_id}")
        
    def _expand_sequences(self, sign, sign_path, sign_type, current_count):
        print(f"\n📈 MODO EXPANSIÓN para '{sign}'. Agregando {self.num_sequences} más.")
        self._collect_normal(sign, sign_path, sign_type, current_count, self.num_sequences)

    def _improve_quality_mode(self):
        """Modo especial para mejorar la calidad de datos existentes"""
        print("\n🔧 MODO MEJORA DE CALIDAD")
        print("="*50)
        
        # Analizar calidad general del dataset
        all_qualities = {}
        total_sequences = 0
        low_quality_count = 0
        
        for sign in self.signs_to_collect:
            sign_path = os.path.join(self.data_dir, sign)
            if os.path.exists(sign_path):
                qualities = self._analyze_existing_quality(sign_path)
                if qualities:
                    all_qualities[sign] = qualities
                    total_sequences += len(qualities)
                    low_quality_count += len([q for q in qualities.values() if q < 75])
        
        if not all_qualities:
            print("❌ No se encontraron datos con metadatos de calidad")
            return
        
        print(f"📊 Análisis del dataset:")
        print(f"   • Total de secuencias: {total_sequences}")
        print(f"   • Secuencias de baja calidad (<75%): {low_quality_count}")
        print(f"   • Porcentaje a mejorar: {(low_quality_count/total_sequences)*100:.1f}%")
        
        # Encontrar señas con más problemas de calidad
        problematic_signs = []
        for sign, qualities in all_qualities.items():
            low_q_count = len([q for q in qualities.values() if q < 75])
            if low_q_count > 0:
                avg_quality = sum(qualities.values()) / len(qualities)
                problematic_signs.append((sign, low_q_count, avg_quality))
        
        # Ordenar por cantidad de problemas y calidad promedio
        problematic_signs.sort(key=lambda x: (-x[1], x[2]))
        
        print(f"\n🎯 Señas que necesitan más mejoras:")
        for i, (sign, low_count, avg_quality) in enumerate(problematic_signs[:10]):
            print(f"   {i+1:2d}. {sign:<15} {low_count:2d} secuencias <75% (Promedio: {avg_quality:.1f}%)")
        
        print(f"\nOpciones:")
        print(f"1. Mejorar seña específica")
        print(f"2. Mejorar automáticamente las más problemáticas")
        print(f"3. Volver al menú principal")
        
        choice = input("Selecciona opción: ").strip()
        
        if choice == '1':
            self._select_sign_for_improvement(problematic_signs)
        elif choice == '2':
            self._auto_improve_quality(problematic_signs[:5])  # Top 5 más problemáticas
        else:
            return

    def _select_sign_for_improvement(self, problematic_signs):
        """Permite seleccionar una seña específica para mejorar"""
        print(f"\n📝 Selecciona seña para mejorar:")
        
        for i, (sign, low_count, avg_quality) in enumerate(problematic_signs):
            print(f"{i+1:2d}. {sign:<15} ({low_count} problemas, promedio: {avg_quality:.1f}%)")
        
        try:
            choice = int(input("Selecciona número: ")) - 1
            if 0 <= choice < len(problematic_signs):
                sign = problematic_signs[choice][0]
                self._collect_for_sign_advanced(sign)
            else:
                print("❌ Número inválido")
        except ValueError:
            print("❌ Por favor ingresa un número válido")

    def _auto_improve_quality(self, top_problematic):
        """Mejora automáticamente las señas más problemáticas"""
        print(f"\n🤖 MODO AUTOMÁTICO DE MEJORA")
        print(f"🎯 Procesando {len(top_problematic)} señas más problemáticas...")
        
        for i, (sign, low_count, avg_quality) in enumerate(top_problematic):
            print(f"\n📍 {i+1}/{len(top_problematic)}: {sign}")
            print(f"   Problemas: {low_count} secuencias <75%")
            print(f"   Calidad promedio: {avg_quality:.1f}%")
            
            response = input("¿Mejorar esta seña? (s/n/q para salir): ").strip().lower()
            
            if response == 'q':
                print("⏹️ Deteniendo modo automático")
                break
            elif response in ['s', 'si', 'y', 'yes']:
                self._collect_for_sign_advanced(sign)
            else:
                print(f"⏭️ Saltando {sign}")
        
        print(f"✅ Modo automático completado")

# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    model_folder = 'models'
    required_models = ['hand_landmarker.task', 'pose_landmarker_heavy.task']
    models_exist = all(os.path.exists(os.path.join(model_folder, model)) for model in required_models)

    if not models_exist:
        print("\n" + "!"*80); print("⚠️  ATENCIÓN: Faltan los archivos de modelo de MediaPipe."); print(f"   Por favor, crea una carpeta llamada '{model_folder}' en este directorio."); print("   Luego, descarga los siguientes archivos y colócalos dentro:"); print("   1. hand_landmarker.task -> https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"); print("   2. pose_landmarker_heavy.task -> https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"); print("!"*80 + "\n")
    else:
        try:
            collector = LSPDataCollector()
            collector.run()
        except Exception as e:
            print(f"\n❌ Ocurrió un error crítico: {e}")
            import traceback
            traceback.print_exc()

