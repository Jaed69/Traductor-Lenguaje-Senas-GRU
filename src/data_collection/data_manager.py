"""
Data Management and Storage - Keras Compatible Format
Maneja el almacenamiento y gestión de datos de secuencias optimizado para Keras/TensorFlow
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any


class DataManager:
    """Gestiona el almacenamiento y metadatos de las secuencias en formato Keras"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.sequences_dir = os.path.join(data_dir, 'sequences')
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        
        # Crear directorios necesarios
        os.makedirs(self.sequences_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Archivo principal para mapeo de etiquetas
        self.labels_file = os.path.join(self.metadata_dir, 'labels_map.json')
        self.dataset_info_file = os.path.join(self.metadata_dir, 'dataset_info.json')
        
        # Cargar o crear mapeo de etiquetas
        self.labels_map = self._load_or_create_labels_map()
        
    def _load_or_create_labels_map(self):
        """Carga o crea el mapeo de etiquetas para Keras"""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Crear mapeo inicial vacío
            labels_map = {
                'sign_to_index': {},
                'index_to_sign': {},
                'num_classes': 0,
                'version': '2.0.0',
                'created': datetime.now().isoformat()
            }
            self._save_labels_map(labels_map)
            return labels_map
    
    def _save_labels_map(self, labels_map):
        """Guarda el mapeo de etiquetas"""
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(labels_map, f, indent=4, ensure_ascii=False)
    
    def add_sign_to_labels(self, sign):
        """Agrega una nueva seña al mapeo de etiquetas"""
        if sign not in self.labels_map['sign_to_index']:
            index = self.labels_map['num_classes']
            self.labels_map['sign_to_index'][sign] = index
            self.labels_map['index_to_sign'][str(index)] = sign
            self.labels_map['num_classes'] += 1
            self.labels_map['last_updated'] = datetime.now().isoformat()
            self._save_labels_map(self.labels_map)
        
        return self.labels_map['sign_to_index'][sign]
      
    def get_collected_sequences_count(self, sign):
        """Cuenta secuencias ya recolectadas para una seña"""
        X_file, y_file = self._get_keras_filenames(sign)
        
        if os.path.exists(X_file):
            try:
                X_data = np.load(X_file)
                return len(X_data)
            except:
                return 0
        return 0
    
    def get_next_sequence_id(self, sign):
        """Obtiene el siguiente ID de secuencia para una seña"""
        return self.get_collected_sequences_count(sign) + 1
    
    def _get_keras_filenames(self, sign):
        """Obtiene los nombres de archivos en formato Keras"""
        X_file = os.path.join(self.sequences_dir, f"{sign}_X.npy")
        y_file = os.path.join(self.sequences_dir, f"{sign}_y.npy") 
        return X_file, y_file
    
    def save_sequence(self, sequence_data, sign, sequence_id, metadata):
        """Guarda una secuencia en formato Keras"""
        # Agregar seña al mapeo de etiquetas
        label_index = self.add_sign_to_labels(sign)
        
        # Obtener archivos existentes
        X_file, y_file = self._get_keras_filenames(sign)
        
        # Cargar datos existentes o crear nuevos
        if os.path.exists(X_file) and os.path.exists(y_file):
            try:
                X_existing = np.load(X_file)
                y_existing = np.load(y_file)
            except:
                X_existing = np.empty((0, sequence_data.shape[0], sequence_data.shape[1]), dtype=np.float32)
                y_existing = np.empty((0,), dtype=np.int32)
        else:
            X_existing = np.empty((0, sequence_data.shape[0], sequence_data.shape[1]), dtype=np.float32)
            y_existing = np.empty((0,), dtype=np.int32)
        
        # Asegurar que sequence_data tenga la forma correcta (1, sequence_length, features)
        if len(sequence_data.shape) == 2:
            sequence_data = sequence_data.reshape(1, sequence_data.shape[0], sequence_data.shape[1])
        elif len(sequence_data.shape) == 1:
            # Si es 1D, necesitamos determinar las dimensiones correctas
            # Asumiendo 60 frames con features variables
            num_features = len(sequence_data)
            if num_features == 60 * 157:  # 60 frames * 157 features
                sequence_data = sequence_data.reshape(1, 60, 157)
            else:
                # Formato dinámico basado en las features del extractor
                features_per_frame = num_features // 60 if num_features >= 60 else num_features
                sequence_data = sequence_data.reshape(1, 60, features_per_frame) if num_features >= 60 else sequence_data.reshape(1, 1, num_features)
        
        # Concatenar nuevos datos
        X_new = np.concatenate([X_existing, sequence_data], axis=0)
        y_new = np.concatenate([y_existing, np.array([label_index], dtype=np.int32)], axis=0)
        
        # Guardar en formato Keras
        np.save(X_file, X_new)
        np.save(y_file, y_new)
        
        # Guardar metadatos individuales
        metadata_file = os.path.join(self.metadata_dir, f"{sign}_{sequence_id}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        # Actualizar info del dataset
        self._update_dataset_info(sign, sequence_id, metadata)
        
        return X_file, y_file, metadata_file
    
    def _update_dataset_info(self, sign, sequence_id, metadata):
        """Actualiza la información general del dataset"""
        dataset_info = self._load_dataset_info()
        
        # Actualizar contadores
        if 'total_sequences' not in dataset_info:
            dataset_info['total_sequences'] = 0
        dataset_info['total_sequences'] += 1
        
        if 'signs' not in dataset_info:
            dataset_info['signs'] = {}
        if sign not in dataset_info['signs']:
            dataset_info['signs'][sign] = {'count': 0, 'last_updated': None}
        
        dataset_info['signs'][sign]['count'] += 1
        dataset_info['signs'][sign]['last_updated'] = datetime.now().isoformat()
        dataset_info['last_updated'] = datetime.now().isoformat()
        
        # Guardar información actualizada
        with open(self.dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    def _load_dataset_info(self):
        """Carga la información del dataset"""
        if os.path.exists(self.dataset_info_file):
            with open(self.dataset_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'version': '2.0.0',
                'created': datetime.now().isoformat(),
                'total_sequences': 0,
                'signs': {},
                'format': 'keras_compatible'
            }
    
    def create_metadata(self, sign, sign_type, hands_info, quality_score, 
                       quality_level, motion_features, issues, collection_mode="NORMAL"):
        """Crea estructura de metadatos para una secuencia"""
        metadata = {
            'sign': sign,
            'sign_type': sign_type,
            'hands_count': hands_info.get('count', 0),
            'handedness': hands_info.get('handedness', []),
            'confidence': hands_info.get('confidence', []),
            'quality_score': quality_score,
            'quality_level': quality_level,
            'motion_features': motion_features.tolist() if hasattr(motion_features, 'tolist') else motion_features,
            'issues': issues,
            'collection_mode': collection_mode,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0'
        }
        return metadata
    
    def load_keras_dataset(self, signs=None):
        """Carga el dataset completo en formato Keras"""
        if signs is None:
            # Cargar todas las señas
            signs = list(self.labels_map['sign_to_index'].keys())
        
        X_all = []
        y_all = []
        
        for sign in signs:
            X_file, y_file = self._get_keras_filenames(sign)
            
            if os.path.exists(X_file) and os.path.exists(y_file):
                try:
                    X_sign = np.load(X_file)
                    y_sign = np.load(y_file)
                    
                    X_all.append(X_sign)
                    y_all.append(y_sign)
                except Exception as e:
                    print(f"Error cargando {sign}: {e}")
                    continue
        
        if X_all:
            X_combined = np.concatenate(X_all, axis=0)
            y_combined = np.concatenate(y_all, axis=0)
            return X_combined, y_combined
        else:
            return np.array([]), np.array([])
    
    def get_keras_dataset_info(self):
        """Obtiene información del dataset en formato Keras"""
        info = {
            'num_classes': self.labels_map['num_classes'],
            'labels_map': self.labels_map,
            'signs_available': [],
            'total_sequences': 0,
            'sequences_per_sign': {}
        }
        
        for sign in self.labels_map['sign_to_index'].keys():
            X_file, y_file = self._get_keras_filenames(sign)
            if os.path.exists(X_file):
                try:
                    X_data = np.load(X_file)
                    count = len(X_data)
                    info['signs_available'].append(sign)
                    info['sequences_per_sign'][sign] = count
                    info['total_sequences'] += count
                except:
                    continue
        
        return info
    
    def load_sequence(self, sign, sequence_id):
        """Carga una secuencia específica (compatible con formato Keras)"""
        X_file, y_file = self._get_keras_filenames(sign)
        
        if os.path.exists(X_file):
            try:
                X_data = np.load(X_file)
                y_data = np.load(y_file)
                
                if sequence_id <= len(X_data):
                    sequence_data = X_data[sequence_id - 1]
                    label = y_data[sequence_id - 1]
                    
                    # Cargar metadatos si existen
                    metadata_file = os.path.join(self.metadata_dir, f"{sign}_{sequence_id}_metadata.json")
                    metadata = None
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    
                    return sequence_data, metadata
            except Exception as e:
                print(f"Error cargando secuencia {sign}_{sequence_id}: {e}")
        
        return None, None
    
    def get_collection_statistics(self):
        """Obtiene estadísticas de la colección de datos (formato Keras)"""
        stats = {
            'total_signs': 0,
            'total_sequences': 0,
            'signs_by_type': {},
            'quality_distribution': {'EXCELENTE': 0, 'BUENA': 0, 'ACEPTABLE': 0, 'REGULAR': 0, 'MALA': 0},
            'completion_status': {},
            'keras_format': True
        }
        
        # Usar información del dataset
        dataset_info = self._load_dataset_info()
        stats['total_sequences'] = dataset_info.get('total_sequences', 0)
        
        # Recorrer señas disponibles
        for sign in self.labels_map['sign_to_index'].keys():
            X_file, y_file = self._get_keras_filenames(sign)
            
            if os.path.exists(X_file):
                try:
                    X_data = np.load(X_file)
                    sequence_count = len(X_data)
                    
                    if sequence_count > 0:
                        stats['total_signs'] += 1
                        stats['completion_status'][sign] = sequence_count
                        
                        # Analizar metadatos para estadísticas
                        for seq_id in range(1, sequence_count + 1):
                            metadata_file = os.path.join(self.metadata_dir, f"{sign}_{seq_id}_metadata.json")
                            
                            if os.path.exists(metadata_file):
                                try:
                                    with open(metadata_file, 'r', encoding='utf-8') as f:
                                        metadata = json.load(f)
                                    
                                    # Estadísticas por tipo de seña
                                    sign_type = metadata.get('sign_type', 'unknown')
                                    if sign_type not in stats['signs_by_type']:
                                        stats['signs_by_type'][sign_type] = 0
                                    stats['signs_by_type'][sign_type] += 1
                                    
                                    # Distribución de calidad
                                    quality_level = metadata.get('quality_level', 'UNKNOWN')
                                    if quality_level in stats['quality_distribution']:
                                        stats['quality_distribution'][quality_level] += 1
                                        
                                except (json.JSONDecodeError, KeyError):
                                    continue
                                    
                except Exception as e:
                    print(f"Error procesando estadísticas para {sign}: {e}")
                    continue
        
        return stats
    
    def export_dataset_summary(self, output_file='dataset_summary.json'):
        """Exporta un resumen completo del dataset en formato Keras"""
        stats = self.get_collection_statistics()
        keras_info = self.get_keras_dataset_info()
        
        summary = {
            'dataset_info': {
                'name': 'LSP Dataset - Keras Compatible',
                'version': '2.0.0',
                'format': 'keras_compatible',
                'creation_date': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'sequences_directory': self.sequences_dir,
                'metadata_directory': self.metadata_dir
            },
            'statistics': stats,
            'keras_info': keras_info,
            'file_structure': self._get_keras_file_structure()
        }
        
        output_path = os.path.join(self.metadata_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        return output_path
    
    def _get_keras_file_structure(self):
        """Obtiene la estructura de archivos del dataset en formato Keras"""
        structure = {
            'sequences': {},
            'metadata_files': [],
            'labels_map': self.labels_file
        }
        
        # Archivos de secuencias por seña
        for sign in self.labels_map['sign_to_index'].keys():
            X_file, y_file = self._get_keras_filenames(sign)
            if os.path.exists(X_file) and os.path.exists(y_file):
                try:
                    X_data = np.load(X_file)
                    structure['sequences'][sign] = {
                        'X_file': os.path.basename(X_file),
                        'y_file': os.path.basename(y_file),
                        'num_sequences': len(X_data),
                        'shape': X_data.shape
                    }
                except Exception as e:
                    structure['sequences'][sign] = {'error': str(e)}
        
        # Archivos de metadatos
        if os.path.exists(self.metadata_dir):
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('_metadata.json')]
            structure['metadata_files'] = metadata_files
        
        return structure
    
    def validate_keras_dataset_integrity(self):
        """Valida la integridad del dataset en formato Keras"""
        issues = []
        
        if not os.path.exists(self.sequences_dir):
            issues.append("Directorio de secuencias no existe")
            return issues
        
        if not os.path.exists(self.metadata_dir):
            issues.append("Directorio de metadatos no existe")
        
        # Validar cada seña
        for sign in self.labels_map['sign_to_index'].keys():
            X_file, y_file = self._get_keras_filenames(sign)
            
            # Verificar que existen ambos archivos
            if not os.path.exists(X_file):
                issues.append(f"Falta archivo X para {sign}: {X_file}")
                continue
            
            if not os.path.exists(y_file):
                issues.append(f"Falta archivo y para {sign}: {y_file}")
                continue
            
            # Verificar que se pueden cargar
            try:
                X_data = np.load(X_file)
                y_data = np.load(y_file)
                
                # Verificar consistencia
                if len(X_data) != len(y_data):
                    issues.append(f"Inconsistencia en {sign}: X={len(X_data)}, y={len(y_data)}")
                
                if X_data.size == 0:
                    issues.append(f"Archivo X vacío para {sign}")
                
                if y_data.size == 0:
                    issues.append(f"Archivo y vacío para {sign}")
                    
            except Exception as e:
                issues.append(f"Error cargando {sign}: {e}")
        
        return issues
    
    def cleanup_keras_dataset(self):
        """Limpia archivos corruptos del dataset Keras"""
        removed_count = 0
        
        for sign in list(self.labels_map['sign_to_index'].keys()):
            X_file, y_file = self._get_keras_filenames(sign)
            should_remove = False
            
            try:
                if os.path.exists(X_file) and os.path.exists(y_file):
                    X_data = np.load(X_file)
                    y_data = np.load(y_file)
                    
                    if X_data.size == 0 or y_data.size == 0 or len(X_data) != len(y_data):
                        should_remove = True
                else:
                    should_remove = True
                    
            except Exception:
                should_remove = True
            
            if should_remove:
                try:
                    if os.path.exists(X_file):
                        os.remove(X_file)
                    if os.path.exists(y_file):
                        os.remove(y_file)
                    
                    # Remover del mapeo de etiquetas
                    if sign in self.labels_map['sign_to_index']:
                        index = self.labels_map['sign_to_index'][sign]
                        del self.labels_map['sign_to_index'][sign]
                        if str(index) in self.labels_map['index_to_sign']:
                            del self.labels_map['index_to_sign'][str(index)]
                        self.labels_map['num_classes'] -= 1
                        self._save_labels_map(self.labels_map)
                    
                    removed_count += 1
                    print(f"Eliminado dataset corrupto para seña: {sign}")
                    
                except Exception as e:
                    print(f"Error eliminando archivos para {sign}: {e}")
        
        return removed_count
