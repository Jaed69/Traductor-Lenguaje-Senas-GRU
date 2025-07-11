"""
Feature Extraction and Processing
Maneja la extracción de características optimizadas para GRU
"""
import numpy as np
from collections import deque


class FeatureExtractor:
    """Extrae y procesa características optimizadas para GRU bidireccional"""
    
    def __init__(self, feature_normalization=True, temporal_smoothing=True):
        self.feature_normalization = feature_normalization
        self.temporal_smoothing = temporal_smoothing
        self.gru_optimized_features = True
        
        # Para cálculo de velocidades temporales
        self.prev_right_hand = None
        self.prev_left_hand = None
        self.prev_pose = None
        
    def normalize_hand_landmarks(self, landmarks_list, handedness):
        """Normaliza landmarks para ser independientes de la mano"""
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

    def extract_advanced_landmarks(self, hand_results, pose_results):
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
                normalized_landmarks = self.normalize_hand_landmarks(hand_landmarks_list, handedness)
                
                # Asignar a posición correspondiente
                if handedness == 'Right' or i == 0:
                    hand_data[0:63] = normalized_landmarks
                    # Calcular velocidad de mano derecha (para GRU temporal)
                    if self.prev_right_hand is not None:
                        velocity_data[0:3] = np.linalg.norm(normalized_landmarks[0:3] - self.prev_right_hand[0:3])
                    self.prev_right_hand = normalized_landmarks
                else:
                    hand_data[63:126] = normalized_landmarks
                    # Calcular velocidad de mano izquierda
                    if self.prev_left_hand is not None:
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
                if self.prev_pose is not None:
                    pose_velocity = np.linalg.norm(pose_data - self.prev_pose)
                    velocity_data = np.append(velocity_data, pose_velocity)
                else:
                    velocity_data = np.append(velocity_data, 0.0)
                self.prev_pose = pose_data

        # Combinar features optimizadas para GRU
        combined_features = np.concatenate([hand_data, pose_data, velocity_data])
        
        # Aplicar normalización si está habilitada
        if self.feature_normalization:
            combined_features = self._normalize_features_for_gru(combined_features)
        
        return combined_features, hands_info

    def _normalize_features_for_gru(self, features):
        """Normaliza features específicamente para GRU bidireccional"""
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
