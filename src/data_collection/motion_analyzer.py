"""
Motion Analysis and Quality Assessment
Calcula métricas de movimiento y evalúa la calidad de las secuencias
"""
import numpy as np

class MotionAnalyzer:
    """Analiza movimiento y calidad de secuencias para GRU."""
    
    def __init__(self, stillness_threshold=0.02, face_proximity_threshold=0.1):
        self.stillness_threshold = stillness_threshold
        self.face_proximity_threshold = face_proximity_threshold
        self.prev_hand_landmarks = None

    def is_user_ready(self, hand_results, pose_results):
        """Verifica si el usuario está en una posición de inicio neutral y lista."""
        # 1. Verificar que las manos están detectadas
        if not hand_results or not hand_results.hand_landmarks:
            self.prev_hand_landmarks = None
            return False, "Manos no detectadas"

        # 2. Verificar que las manos están relativamente quietas
        current_hand_landmarks = hand_results.hand_landmarks
        if self.prev_hand_landmarks:
            movement = 0
            for i in range(len(current_hand_landmarks)):
                if i < len(self.prev_hand_landmarks):
                    # Calcular movimiento promedio de todos los landmarks de la mano
                    curr_pts = np.array([(lm.x, lm.y, lm.z) for lm in current_hand_landmarks[i]])
                    prev_pts = np.array([(lm.x, lm.y, lm.z) for lm in self.prev_hand_landmarks[i]])
                    movement += np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1))
            
            avg_movement = movement / len(current_hand_landmarks)
            if avg_movement > self.stillness_threshold:
                self.prev_hand_landmarks = current_hand_landmarks
                return False, f"Mantén las manos más quietas (mov: {avg_movement:.3f})"

        self.prev_hand_landmarks = current_hand_landmarks

        # 3. Verificar que las manos no están cerca de la cara
        if pose_results and pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks[0]
            face_landmarks = [pose_landmarks[i] for i in [0, 1, 4, 7, 8, 9, 10]] # Nariz, ojos, orejas
            
            for hand_landmarks in current_hand_landmarks:
                wrist = hand_landmarks[0]
                for face_lm in face_landmarks:
                    distance = np.linalg.norm([wrist.x - face_lm.x, wrist.y - face_lm.y])
                    if distance < self.face_proximity_threshold:
                        return False, "Aleja las manos del rostro"

        # 4. Verificar que los brazos están en una posición relajada (opcional, más complejo)
        # Podríamos verificar que los codos están por debajo de los hombros, etc.

        return True, "¡Listo!"

    def calculate_motion_features(self, sequence_data):
        """Calcula un conjunto de métricas de movimiento optimizadas para GRU."""
        if len(sequence_data) < 5:
            return np.zeros(20)

        hand_sequence = sequence_data[:, :126]
        frame_movements = [np.linalg.norm(hand_sequence[i] - hand_sequence[i-1]) for i in range(1, len(hand_sequence))]
        if not frame_movements: return np.zeros(20)

        accelerations = [abs(frame_movements[i] - frame_movements[i-1]) for i in range(1, len(frame_movements))]
        jerk_values = [abs(accelerations[i] - accelerations[i-1]) for i in range(1, len(accelerations))]

        # Métricas de movimiento, estabilidad, consistencia, etc.
        metrics = [
            sum(frame_movements),                                       # 1. Movimiento total
            np.mean(frame_movements),                                   # 2. Velocidad promedio
            max(frame_movements),                                       # 3. Velocidad máxima
            np.var(frame_movements),                                    # 4. Varianza de velocidad
            np.mean(accelerations) if accelerations else 0,             # 5. Aceleración promedio
            max(accelerations) if accelerations else 0,                 # 6. Aceleración máxima
            np.mean(jerk_values) if jerk_values else 0,                 # 7. Jerk promedio (suavidad)
            np.linalg.norm(hand_sequence[-1] - hand_sequence[0]),      # 8. Desplazamiento neto
            1.0 / (1.0 + np.var(frame_movements)),                      # 9. Consistencia temporal
            1.0 / (1.0 + np.mean(jerk_values)) if jerk_values else 1.0, # 10. Suavidad del movimiento
        ]
        
        # Rellenar hasta 20 métricas si es necesario
        metrics.extend([0.0] * (20 - len(metrics)))
        return np.array(metrics)

    def evaluate_sequence_quality(self, sequence_data, motion_features, sign_type):
        """Evalúa la calidad de una secuencia basado en métricas de movimiento."""
        quality_score = 100.0
        issues = []

        # 1. Completitud de los datos
        completeness = np.count_nonzero(np.any(sequence_data[:, :126] != 0, axis=1)) / len(sequence_data)
        if completeness < 0.9:
            quality_score -= 30
            issues.append(f"Datos de mano incompletos ({completeness:.1%})")

        # 2. Análisis de movimiento por tipo de seña
        avg_movement = motion_features[1]
        if 'static' in sign_type and avg_movement > 0.015:
            quality_score -= 25
            issues.append("Demasiado movimiento para una seña estática")
        elif 'dynamic' in sign_type and avg_movement < 0.008:
            quality_score -= 25
            issues.append("Movimiento insuficiente para una seña dinámica")

        # 3. Suavidad del movimiento (bajo jerk)
        jerk_avg = motion_features[6]
        if jerk_avg > 0.05:
            quality_score -= 20
            issues.append(f"Movimiento brusco o tembloroso (jerk: {jerk_avg:.3f})")

        # 4. Consistencia
        temporal_consistency = motion_features[8]
        if temporal_consistency < 0.7:
            quality_score -= 15
            issues.append("El ritmo del movimiento fue inconsistente")

        quality_score = max(0, min(100, quality_score))
        
        if quality_score >= 90: quality_level = "EXCELENTE"
        elif quality_score >= 75: quality_level = "BUENA"
        elif quality_score >= 60: quality_level = "ACEPTABLE"
        else: quality_level = "MALA"
        
        return quality_score, quality_level, issues
