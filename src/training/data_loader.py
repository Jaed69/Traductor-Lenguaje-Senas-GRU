"""
Data Loader - Gestor Eficiente de Datos HDF5
Carga y preprocesamiento de datos para entrenamiento de modelos GRU

Autor: LSP Team
Versión: 2.0 - Julio 2025
"""

import os
import h5py
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


class HDF5DataLoader:
    """
    Gestor eficiente de datos HDF5 para entrenamiento de modelos GRU
    """
    
    def __init__(self, data_path: str = "data", sequence_length: int = 60):
        """
        Inicializa el gestor de datos
        
        Args:
            data_path: Ruta a la carpeta de datos
            sequence_length: Longitud de secuencias para el modelo
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.sequences_file = os.path.join(data_path, "sequences.h5")
        self.metadata_path = os.path.join(data_path, "metadata")
        
        # Variables de estado
        self.label_encoder = LabelEncoder()
        self.dataset_info = None
        self.labels_map = None
        
        print("🗃️ Inicializando Gestor de Datos HDF5")
        print(f"   📁 Ruta de datos: {self.data_path}")
        print(f"   📏 Longitud de secuencia: {self.sequence_length}")
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadatos del dataset"""
        try:
            # Cargar información del dataset
            dataset_info_path = os.path.join(self.metadata_path, "dataset_info.json")
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    self.dataset_info = json.load(f)
                print(f"✅ Dataset info cargado - {self.dataset_info.get('total_sequences', 0)} secuencias")
            
            # Cargar mapeo de etiquetas
            labels_map_path = os.path.join(self.metadata_path, "labels_map.json")
            if os.path.exists(labels_map_path):
                with open(labels_map_path, 'r', encoding='utf-8') as f:
                    self.labels_map = json.load(f)
                print(f"✅ Mapeo de etiquetas cargado - {len(self.labels_map)} clases")
            
        except Exception as e:
            print(f"⚠️ Error cargando metadatos: {e}")
    
    def check_data_availability(self) -> bool:
        """
        Verifica la disponibilidad de datos de entrenamiento
        
        Returns:
            bool: True si los datos están disponibles
        """
        print("\n🔍 VERIFICANDO DISPONIBILIDAD DE DATOS...")
        
        if not os.path.exists(self.sequences_file):
            print(f"❌ Archivo HDF5 no encontrado: {self.sequences_file}")
            return False
        
        try:
            with h5py.File(self.sequences_file, 'r') as f:
                groups = list(f.keys())
                print(f"✅ Archivo HDF5 encontrado con {len(groups)} grupos")
                
                total_sequences = 0
                for group_name in groups:
                    group = f[group_name]
                    if isinstance(group, h5py.Group) and 'sequences' in group:
                        sequences = group['sequences']
                        if isinstance(sequences, h5py.Dataset):
                            total_sequences += sequences.shape[0]
                            print(f"   📋 {group_name}: {sequences.shape[0]} secuencias")
                
                print(f"📊 Total de secuencias disponibles: {total_sequences}")
                return total_sequences > 0
                
        except Exception as e:
            print(f"❌ Error leyendo archivo HDF5: {e}")
            return False
    
    def load_dataset(self, test_size: float = 0.2, val_size: float = 0.1, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga el dataset completo desde HDF5 y lo divide en train/val/test
        
        Args:
            test_size: Proporción de datos para test
            val_size: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tuple con (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n📂 CARGANDO DATASET DESDE HDF5...")
        
        if not self.check_data_availability():
            raise ValueError("Datos no disponibles para entrenamiento")
        
        # Cargar todas las secuencias y etiquetas
        X_data = []
        y_data = []
        
        with h5py.File(self.sequences_file, 'r') as f:
            for sign_name in f.keys():
                group = f[sign_name]
                
                if isinstance(group, h5py.Group) and 'sequences' in group:
                    sequences_dataset = group['sequences']
                    if isinstance(sequences_dataset, h5py.Dataset):
                        sequences = sequences_dataset[:]
                        
                        # Obtener etiquetas
                        if 'labels' in group and isinstance(group['labels'], h5py.Dataset):
                            labels_dataset = group['labels']
                            labels = labels_dataset[:]
                        else:
                            labels = [sign_name] * len(sequences)
                        
                        # Redimensionar secuencias si es necesario
                        sequences = self._adjust_sequence_length(sequences)
                        
                        X_data.append(sequences)
                        y_data.extend(labels)
                        
                        print(f"   ✅ {sign_name}: {len(sequences)} secuencias cargadas")
        
        # Concatenar todos los datos
        X = np.vstack(X_data)
        y = np.array(y_data)
        
        print(f"\n📊 DATOS CARGADOS:")
        print(f"   🔢 Forma de X: {X.shape}")
        print(f"   🔢 Forma de y: {y.shape}")
        print(f"   📋 Clases únicas: {len(np.unique(y))}")
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y)
        
        # División estratificada
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        # División train/validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_train_val
        )
        
        print(f"\n📈 DIVISIÓN DE DATOS:")
        print(f"   🚂 Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   ✅ Validación: {X_val.shape[0]} muestras ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"   🧪 Test: {X_test.shape[0]} muestras ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _adjust_sequence_length(self, sequences: np.ndarray) -> np.ndarray:
        """
        Ajusta la longitud de las secuencias al tamaño requerido
        
        Args:
            sequences: Array de secuencias
            
        Returns:
            Secuencias ajustadas
        """
        current_length = sequences.shape[1]
        
        if current_length == self.sequence_length:
            return sequences
        
        elif current_length > self.sequence_length:
            # Truncar desde el centro
            start_idx = (current_length - self.sequence_length) // 2
            return sequences[:, start_idx:start_idx + self.sequence_length, :]
        
        else:
            # Padding con repetición del último frame
            padding_needed = self.sequence_length - current_length
            last_frame = sequences[:, -1:, :]
            padding = np.repeat(last_frame, padding_needed, axis=1)
            return np.concatenate([sequences, padding], axis=1)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del dataset
        
        Returns:
            Diccionario con estadísticas
        """
        print("\n📊 CALCULANDO ESTADÍSTICAS DEL DATASET...")
        
        stats = {
            'total_sequences': 0,
            'signs': {},
            'sequence_length_stats': {},
            'feature_stats': {},
            'class_distribution': {},
            'data_quality': {}
        }
        
        if not os.path.exists(self.sequences_file):
            return stats
        
        try:
            with h5py.File(self.sequences_file, 'r') as f:
                sequence_lengths = []
                feature_dimensions = []
                
                for sign_name in f.keys():
                    group = f[sign_name]
                    
                    if isinstance(group, h5py.Group) and 'sequences' in group:
                        sequences_dataset = group['sequences']
                        if isinstance(sequences_dataset, h5py.Dataset):
                            count = sequences_dataset.shape[0]
                            
                            stats['signs'][sign_name] = {
                                'count': count,
                                'shape': sequences_dataset.shape,
                                'dtype': str(sequences_dataset.dtype)
                            }
                            
                            stats['total_sequences'] += count
                            sequence_lengths.extend([sequences_dataset.shape[1]] * count)
                            feature_dimensions.append(sequences_dataset.shape[2])
                            
                            # Distribución de clases
                            stats['class_distribution'][sign_name] = count
                
                # Estadísticas de longitud de secuencia
                if sequence_lengths:
                    stats['sequence_length_stats'] = {
                        'mean': float(np.mean(sequence_lengths)),
                        'std': float(np.std(sequence_lengths)),
                        'min': int(np.min(sequence_lengths)),
                        'max': int(np.max(sequence_lengths)),
                        'target': self.sequence_length
                    }
                
                # Estadísticas de características
                if feature_dimensions:
                    stats['feature_stats'] = {
                        'dimensions': feature_dimensions,
                        'consistent': len(set(feature_dimensions)) == 1
                    }
                
                # Calidad de datos
                if stats['class_distribution']:
                    stats['data_quality'] = {
                        'consistent_shapes': len(set(feature_dimensions)) == 1,
                        'min_samples_per_class': min(stats['class_distribution'].values()),
                        'max_samples_per_class': max(stats['class_distribution'].values()),
                        'balanced_threshold': 0.7
                    }
                
        except Exception as e:
            print(f"❌ Error calculando estadísticas: {e}")
        
        return stats
    
    def normalize_data(self, X_train: np.ndarray, X_val: np.ndarray, 
                      X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Normaliza los datos usando estadísticas del conjunto de entrenamiento
        
        Args:
            X_train, X_val, X_test: Conjuntos de datos
            
        Returns:
            Datos normalizados y estadísticas de normalización
        """
        print("\n🔧 NORMALIZANDO DATOS...")
        
        # Calcular estadísticas solo del conjunto de entrenamiento
        train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
        train_std = np.std(X_train, axis=(0, 1), keepdims=True)
        
        # Evitar división por cero
        train_std = np.where(train_std == 0, 1, train_std)
        
        # Normalizar todos los conjuntos
        X_train_norm = (X_train - train_mean) / train_std
        X_val_norm = (X_val - train_mean) / train_std
        X_test_norm = (X_test - train_mean) / train_std
        
        normalization_stats = {
            'mean': train_mean,
            'std': train_std,
            'method': 'z-score',
            'computed_on': 'train_set'
        }
        
        print(f"   ✅ Normalización completada (μ={train_mean.mean():.4f}, σ={train_std.mean():.4f})")
        
        return X_train_norm, X_val_norm, X_test_norm, normalization_stats
    
    def save_preprocessing_info(self, normalization_stats: Dict[str, Any], 
                               output_path: Optional[str] = None):
        """
        Guarda información de preprocesamiento para uso en inferencia
        
        Args:
            normalization_stats: Estadísticas de normalización
            output_path: Ruta donde guardar la información
        """
        if output_path is None:
            output_path = os.path.join(self.metadata_path, "preprocessing_info.json")
        
        preprocessing_info = {
            'sequence_length': self.sequence_length,
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'normalization': {
                'mean': normalization_stats['mean'].tolist(),
                'std': normalization_stats['std'].tolist(),
                'method': normalization_stats['method']
            },
            'created': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(preprocessing_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Información de preprocesamiento guardada en: {output_path}")
    
    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calcula pesos de clase para manejar desbalance
        
        Args:
            y_train: Etiquetas de entrenamiento
            
        Returns:
            Diccionario con pesos por clase
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        class_weights = dict(zip(classes, weights))
        
        print(f"⚖️ Pesos de clase calculados para {len(classes)} clases")
        for class_id, weight in class_weights.items():
            class_name = self.label_encoder.inverse_transform([class_id])[0]
            print(f"   {class_name}: {weight:.3f}")
        
        return class_weights


if __name__ == "__main__":
    # Ejemplo de uso
    loader = HDF5DataLoader()
    
    if loader.check_data_availability():
        stats = loader.get_data_statistics()
        print("\n📊 Estadísticas del dataset:")
        print(f"   Total de secuencias: {stats['total_sequences']}")
        print(f"   Número de señas: {len(stats['signs'])}")
        
        # Ejemplo de carga de datos
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = loader.load_dataset()
            X_train_norm, X_val_norm, X_test_norm, norm_stats = loader.normalize_data(
                X_train, X_val, X_test
            )
            
            class_weights = loader.get_class_weights(y_train)
            loader.save_preprocessing_info(norm_stats)
            
            print("✅ Dataset cargado y preprocesado exitosamente")
            
        except Exception as e:
            print(f"❌ Error en el ejemplo: {e}")
