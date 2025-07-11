"""
Motion Analysis and Quality Assessment
Calcula métricas de movimiento y evalúa la calidad de las secuencias
"""
import numpy as np


class MotionAnalyzer:
    """Analiza movimiento y calidad de secuencias para GRU"""
    
    def __init__(self):
        pass
    
    def calculate_motion_features(self, sequence_data):
        """Calcula métricas de movimiento optimizadas para GRU bidireccional"""
        if len(sequence_data) < 5:
            return np.zeros(20)
        
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
            try:
                from scipy import signal
                # Buscar periodicidad en el movimiento
                autocorr = signal.correlate(frame_movements, frame_movements, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                periodicity = np.max(autocorr[1:]) / autocorr[0] if len(autocorr) > 1 else 0.0
            except ImportError:
                # Si scipy no está disponible, usar una aproximación simple
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
    
    def evaluate_sequence_quality(self, sequence_data, motion_features, sign_type):
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
                issues.append("Movimiento insuficiente (puede ser ruido)")
                
            # Consistencia temporal más importante en señas estáticas
            if temporal_consistency < 0.7:
                quality_score -= 15
                issues.append("Inconsistencia temporal en seña estática")
                
        elif 'dynamic' in sign_type:
            # Para señas dinámicas: movimiento fluido y progresivo
            if avg_movement < 0.008:
                quality_score -= 20
                issues.append("Movimiento insuficiente para seña dinámica")
            elif avg_movement > 0.05:
                quality_score -= 15
                issues.append("Movimiento excesivo (puede ser errático)")
                
            # Suavidad más importante en señas dinámicas
            if smoothness < 0.5:
                quality_score -= 15
                issues.append("Movimiento no fluido en seña dinámica")
        
        # 3. Estabilidad inicial y final (crítico para GRU temporal)
        if len(motion_features) > 15:
            start_stability = motion_features[15]
            end_stability = motion_features[16]
            
            if start_stability < 0.8:
                quality_score -= 10
                issues.append("Inicio inestable")
            if end_stability < 0.8:
                quality_score -= 10
                issues.append("Final inestable")
        
        # 4. Coordinación entre manos (importante para GRU bidireccional)
        if len(motion_features) > 14:
            hand_coordination = motion_features[14]
            if 'two_hands' in sign_type and hand_coordination < 0.3:
                quality_score -= 15
                issues.append("Coordinación entre manos deficiente")
        
        # 5. Análisis de complejidad gestual
        if len(motion_features) > 17:
            gesture_complexity = motion_features[17]
            if gesture_complexity > 0.5:
                quality_score -= 5
                issues.append("Gesto muy complejo (puede ser ruido)")
        
        # 6. Evaluación de jerk y aceleración (suavidad para GRU)
        jerk_avg = motion_features[11] if len(motion_features) > 11 else 0.0
        acceleration_avg = motion_features[6] if len(motion_features) > 6 else 0.0
        
        if jerk_avg > 0.05:
            quality_score -= 10
            issues.append("Movimiento no suave (jerk alto)")
        if acceleration_avg > 0.03:
            quality_score -= 8
            issues.append("Aceleración excesiva")
        
        # 7. Bonus por características ideales para GRU
        if temporal_consistency > 0.9:
            quality_score += 5  # Bonus por alta consistencia temporal
        
        # Determinar nivel de calidad
        quality_score = max(0, min(100, quality_score))
        
        if quality_score >= 90:
            quality_level = "EXCELENTE"
        elif quality_score >= 80:
            quality_level = "BUENA"
        elif quality_score >= 70:
            quality_level = "ACEPTABLE"
        elif quality_score >= 60:
            quality_level = "REGULAR"
        else:
            quality_level = "MALA"
        
        return quality_score, quality_level, issues
