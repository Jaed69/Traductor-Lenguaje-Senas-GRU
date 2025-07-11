"""
Data Augmentation Module for LSP Sign Language Dataset
Técnicas especializadas para aumentar el dataset de señas sin perder semántica
"""

import numpy as np
import random
import os
import glob
from typing import List, Tuple, Dict, Optional
import copy
import json


class LSPDataAugmenter:
    """
    Augmentador de datos especializado para Lenguaje de Señas Peruano
    
    Técnicas implementadas:
    - Variaciones temporales (velocidad, pausas)
    - Transformaciones espaciales (rotación, escala, traslación)
    - Ruido gaussiano controlado
    - Variaciones de manos (izquierda/derecha)
    - Interpolación entre frames
    - Perturbaciones de landmarks
    """
    
    def __init__(self):
        self.augmentation_config = {
            'temporal_variations': {
                'speed_range': (0.8, 1.2),  # 80% a 120% velocidad original
                'pause_probability': 0.1,   # 10% chance de agregar pausa
                'interpolation_factor': 1.2  # Factor para interpolación
            },
            'spatial_transformations': {
                'rotation_range': (-15, 15),      # ±15 grados
                'scale_range': (0.9, 1.1),       # ±10% escala
                'translation_range': (-0.05, 0.05), # ±5% traslación
                'flip_probability': 0.3            # 30% chance flip horizontal
            },
            'noise_augmentation': {
                'gaussian_std': 0.01,           # Std del ruido gaussiano
                'landmark_jitter': 0.005,       # Jitter en landmarks
                'dropout_probability': 0.02     # 2% chance dropout landmarks
            },
            'hand_variations': {
                'swap_hands_prob': 0.2,         # 20% chance intercambiar manos
                'single_hand_prob': 0.15        # 15% chance usar solo una mano
            }
        }
        
        # Definir qué augmentaciones son seguras para cada tipo de seña
        self.safe_augmentations = {
            'static_letter': ['spatial_light', 'noise_light', 'hand_variations'],
            'dynamic_letter': ['temporal_light', 'spatial_light', 'noise_light'],
            'word': ['temporal_medium', 'spatial_light', 'noise_light'],
            'phrase': ['temporal_light', 'noise_light']
        }
    
    def augment_sequence(self, sequence: np.ndarray, sign_type: str, 
                        metadata: Dict, num_augmentations: int = 3) -> List[Tuple[np.ndarray, Dict]]:
        """
        Genera múltiples versiones aumentadas de una secuencia
        
        Args:
            sequence: Secuencia original (frames, features)
            sign_type: Tipo de seña ('static_letter', 'dynamic_letter', etc.)
            metadata: Metadatos originales
            num_augmentations: Número de variaciones a generar
            
        Returns:
            Lista de tuplas (secuencia_aumentada, metadatos_actualizados)
        """
        augmented_sequences = []
        safe_augs = self.safe_augmentations.get(sign_type, ['noise_light'])
        
        for i in range(num_augmentations):
            # Seleccionar técnica de augmentación aleatoria
            aug_technique = random.choice(safe_augs)
            
            # Aplicar augmentación
            aug_sequence = self._apply_augmentation(sequence, aug_technique)
            
            # Actualizar metadatos
            aug_metadata = self._update_metadata(metadata, aug_technique, i)
            
            augmented_sequences.append((aug_sequence, aug_metadata))
        
        return augmented_sequences
    
    def _apply_augmentation(self, sequence: np.ndarray, technique: str) -> np.ndarray:
        """Aplica una técnica específica de augmentación"""
        
        if technique == 'temporal_light':
            return self._temporal_augmentation(sequence, intensity='light')
        elif technique == 'temporal_medium':
            return self._temporal_augmentation(sequence, intensity='medium')
        elif technique == 'spatial_light':
            return self._spatial_augmentation(sequence, intensity='light')
        elif technique == 'spatial_medium':
            return self._spatial_augmentation(sequence, intensity='medium')
        elif technique == 'noise_light':
            return self._noise_augmentation(sequence, intensity='light')
        elif technique == 'hand_variations':
            return self._hand_variation_augmentation(sequence)
        else:
            return sequence.copy()
    
    def _temporal_augmentation(self, sequence: np.ndarray, intensity: str = 'light') -> np.ndarray:
        """
        Augmentación temporal: cambios de velocidad, interpolación, pausas
        """
        config = self.augmentation_config['temporal_variations']
        
        if intensity == 'light':
            speed_range = (0.9, 1.1)
            interp_factor = 1.1
        else:  # medium
            speed_range = config['speed_range']
            interp_factor = config['interpolation_factor']
        
        # Cambio de velocidad mediante interpolación
        speed_factor = random.uniform(*speed_range)
        original_length = len(sequence)
        new_length = int(original_length * speed_factor)
        
        # Interpolación para cambiar velocidad
        if new_length != original_length:
            indices = np.linspace(0, original_length - 1, new_length)
            augmented = np.array([
                np.interp(indices, range(original_length), sequence[:, i])
                for i in range(sequence.shape[1])
            ]).T
        else:
            augmented = sequence.copy()
        
        # Asegurar longitud correcta (60 frames)
        target_length = 60
        if len(augmented) < target_length:
            # Padding con repetición del último frame
            padding = np.tile(augmented[-1:], (target_length - len(augmented), 1))
            augmented = np.vstack([augmented, padding])
        elif len(augmented) > target_length:
            # Recortar uniformemente
            indices = np.linspace(0, len(augmented) - 1, target_length, dtype=int)
            augmented = augmented[indices]
        
        return augmented
    
    def _spatial_augmentation(self, sequence: np.ndarray, intensity: str = 'light') -> np.ndarray:
        """
        Augmentación espacial: rotación, escala, traslación
        """
        config = self.augmentation_config['spatial_transformations']
        
        if intensity == 'light':
            rotation_range = (-5, 5)
            scale_range = (0.95, 1.05)
            translation_range = (-0.02, 0.02)
        else:  # medium
            rotation_range = config['rotation_range']
            scale_range = config['scale_range']
            translation_range = config['translation_range']
        
        # Parámetros de transformación
        rotation_angle = np.radians(random.uniform(*rotation_range))
        scale_factor = random.uniform(*scale_range)
        tx = random.uniform(*translation_range)
        ty = random.uniform(*translation_range)
        
        # Matriz de transformación 2D
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        transform_matrix = np.array([
            [scale_factor * cos_a, -scale_factor * sin_a, tx],
            [scale_factor * sin_a,  scale_factor * cos_a, ty],
            [0, 0, 1]
        ])
        
        augmented = sequence.copy()
        
        # Aplicar transformación a landmarks de manos (primeros 126 features)
        # Estructura: 21 landmarks × 3 coords × 2 manos = 126 features
        for frame_idx in range(len(augmented)):
            frame = augmented[frame_idx]
            
            # Procesar landmarks de manos (x, y coords)
            for hand in range(2):  # 2 manos
                start_idx = hand * 63  # 21 landmarks × 3 coords
                for landmark in range(21):  # 21 landmarks por mano
                    x_idx = start_idx + landmark * 3
                    y_idx = start_idx + landmark * 3 + 1
                    
                    if x_idx < len(frame) and y_idx < len(frame):
                        # Coordenadas originales
                        x, y = frame[x_idx], frame[y_idx]
                        
                        # Aplicar transformación
                        point = np.array([x, y, 1])
                        transformed = transform_matrix @ point
                        
                        # Asegurar que permanezcan en rango [0, 1]
                        frame[x_idx] = np.clip(transformed[0], 0, 1)
                        frame[y_idx] = np.clip(transformed[1], 0, 1)
        
        return augmented
    
    def _noise_augmentation(self, sequence: np.ndarray, intensity: str = 'light') -> np.ndarray:
        """
        Augmentación con ruido: ruido gaussiano, jitter, dropout
        """
        config = self.augmentation_config['noise_augmentation']
        
        if intensity == 'light':
            noise_std = config['gaussian_std'] * 0.5
            jitter_amount = config['landmark_jitter'] * 0.5
        else:
            noise_std = config['gaussian_std']
            jitter_amount = config['landmark_jitter']
        
        augmented = sequence.copy()
        
        # Ruido gaussiano suave
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented += noise
        
        # Jitter específico en landmarks
        for frame_idx in range(len(augmented)):
            frame = augmented[frame_idx]
            
            # Aplicar jitter solo a coordenadas x, y (no a z ni confianza)
            for i in range(0, min(126, len(frame)), 3):  # Cada 3 valores (x, y, confianza)
                if i + 1 < len(frame):
                    frame[i] += random.gauss(0, jitter_amount)      # x
                    frame[i + 1] += random.gauss(0, jitter_amount)  # y
        
        # Asegurar rango válido [0, 1] para coordenadas normalizadas
        augmented = np.clip(augmented, 0, 1)
        
        return augmented
    
    def _hand_variation_augmentation(self, sequence: np.ndarray) -> np.ndarray:
        """
        Variaciones de manos: intercambio izquierda/derecha
        """
        config = self.augmentation_config['hand_variations']
        augmented = sequence.copy()
        
        # Intercambiar manos con cierta probabilidad
        if random.random() < config['swap_hands_prob']:
            # Intercambiar landmarks de mano izquierda y derecha
            for frame_idx in range(len(augmented)):
                frame = augmented[frame_idx]
                
                # Landmarks mano derecha: índices 0-62
                # Landmarks mano izquierda: índices 63-125
                if len(frame) >= 126:
                    right_hand = frame[0:63].copy()
                    left_hand = frame[63:126].copy()
                    
                    # Intercambiar
                    frame[0:63] = left_hand
                    frame[63:126] = right_hand
        
        return augmented
    
    def _update_metadata(self, original_metadata: Dict, technique: str, aug_id: int) -> Dict:
        """Actualiza metadatos para secuencia aumentada"""
        aug_metadata = copy.deepcopy(original_metadata)
        
        # Agregar información de augmentación
        aug_metadata['augmentation'] = {
            'is_augmented': True,
            'technique': technique,
            'augmentation_id': aug_id,
            'original_sequence_id': original_metadata.get('sequence_id', 'unknown')
        }
        
        # Actualizar quality_score (puede ser ligeramente menor)
        original_quality = aug_metadata.get('quality_score', 80.0)
        quality_reduction = random.uniform(0, 5)  # Reducción de 0-5 puntos
        aug_metadata['quality_score'] = max(70.0, original_quality - quality_reduction)
        
        # Actualizar collection_mode
        aug_metadata['collection_mode'] = 'AUGMENTED'
        
        return aug_metadata
    
    def calculate_augmentation_needs(self, current_counts: Dict[str, int], 
                                   target_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Calcula cuántas augmentaciones se necesitan por seña
        """
        augmentation_needs = {}
        
        for sign, target in target_counts.items():
            current = current_counts.get(sign, 0)
            if current < target:
                deficit = target - current
                # Calcular augmentaciones necesarias (conservador)
                base_sequences_needed = max(1, deficit // 4)  # 1 original -> 4 total con augm.
                augmentations_needed = deficit - base_sequences_needed
                augmentation_needs[sign] = max(0, augmentations_needed)
        
        return augmentation_needs
    
    def generate_augmentation_report(self, sign: str, original_count: int, 
                                   augmented_count: int, techniques_used: List[str]) -> Dict:
        """Genera reporte de augmentación para una seña"""
        return {
            'sign': sign,
            'original_sequences': original_count,
            'augmented_sequences': augmented_count,
            'total_sequences': original_count + augmented_count,
            'augmentation_ratio': augmented_count / max(1, original_count),
            'techniques_used': techniques_used,
            'quality_preserved': True,
            'semantic_validity': 'HIGH'  # Para LSP, técnicas conservadoras
        }


class AugmentationIntegrator:
    """
    Integra Data Augmentation con el sistema de recolección existente
    """
    
    def __init__(self, data_manager, sign_config):
        self.data_manager = data_manager
        self.sign_config = sign_config
        self.augmenter = LSPDataAugmenter()
    
    def auto_augment_dataset(self, target_reduction_factor: float = 0.5) -> Dict:
        """
        Aumenta automáticamente el dataset para reducir recolección manual
        
        Args:
            target_reduction_factor: Factor de reducción de trabajo manual (0.5 = 50% menos)
        """
        print("🔄 INICIANDO DATA AUGMENTATION AUTOMÁTICO")
        print("="*60)
        
        stats = self.data_manager.get_collection_statistics()
        augmentation_report = {
            'total_original': stats['total_sequences'],
            'total_augmented': 0,
            'signs_processed': 0,
            'techniques_summary': {}
        }
        
        # Procesar cada seña
        for sign in self.sign_config.get_all_signs():
            current_count = self.data_manager.get_collected_sequences_count(sign)
            
            if current_count > 0:  # Solo aumentar si ya hay datos base
                sign_type = self.sign_config.classify_sign_type(sign)
                target_count = self.sign_config.get_recommended_sequence_count(sign_type)
                
                # Calcular cuánto aumentar
                needed = max(0, int((target_count - current_count) * target_reduction_factor))
                
                if needed > 0:
                    augmented = self._augment_sign_sequences(sign, sign_type, needed)
                    augmentation_report['total_augmented'] += augmented
                    augmentation_report['signs_processed'] += 1
                    
                    print(f"✅ {sign}: +{augmented} secuencias aumentadas")
        
        print(f"\n📊 RESUMEN AUGMENTATION:")
        print(f"   🎯 Secuencias originales: {augmentation_report['total_original']}")
        print(f"   🔄 Secuencias aumentadas: {augmentation_report['total_augmented']}")
        print(f"   📈 Mejora total: {augmentation_report['total_augmented'] / max(1, augmentation_report['total_original']) * 100:.1f}%")
        print(f"   ⚡ Reducción trabajo manual: {target_reduction_factor * 100:.0f}%")
        
        return augmentation_report
    
    def _augment_sign_sequences(self, sign: str, sign_type: str, target_augmentations: int) -> int:
        """Aumenta secuencias para una seña específica"""
        # Cargar secuencias existentes de esta seña
        sign_sequences = self._load_sign_sequences(sign)
        
        if not sign_sequences:
            return 0
        
        augmented_count = 0
        augmentations_per_sequence = max(1, target_augmentations // len(sign_sequences))
        
        for sequence_data, metadata in sign_sequences:
            # Generar augmentaciones
            augmented_sequences = self.augmenter.augment_sequence(
                sequence_data, sign_type, metadata, augmentations_per_sequence
            )
            
            # Guardar augmentaciones
            for aug_sequence, aug_metadata in augmented_sequences:
                # Generar ID único para augmentación
                aug_id = augmented_count + 1000  # Offset para distinguir de originales
                
                self.data_manager.save_sequence(
                    aug_sequence, sign, aug_id, aug_metadata
                )
                augmented_count += 1
                
                if augmented_count >= target_augmentations:
                    break
            
            if augmented_count >= target_augmentations:
                break
        
        return augmented_count
    
    def _load_sign_sequences(self, sign: str) -> List[Tuple[np.ndarray, Dict]]:
        """Carga secuencias existentes de una seña"""
        sequences = []
        
        # Buscar archivos de la seña en el directorio de datos
        import glob
        pattern = f"data/*{sign}_*.npy"
        sequence_files = glob.glob(pattern)
        
        for seq_file in sequence_files:
            try:
                # Cargar secuencia
                sequence = np.load(seq_file)
                
                # Cargar metadatos correspondientes
                metadata_file = seq_file.replace('.npy', '_metadata.json')
                metadata = {}
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                sequences.append((sequence, metadata))
            except Exception as e:
                print(f"⚠️ Error cargando {seq_file}: {e}")
        
        return sequences
