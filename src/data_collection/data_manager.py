"""
Data Management and Storage - HDF5 Backend
Maneja el almacenamiento y gestión de datos de secuencias optimizado para Keras/TensorFlow usando HDF5.
"""
import os
import json
import numpy as np
import h5py
from datetime import datetime
from typing import List, Tuple, Dict, Any

class DataManager:
    """Gestiona el almacenamiento y metadatos de las secuencias en formato HDF5"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        # El directorio 'sequences' ya no es necesario, se usará un único archivo HDF5
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        
        # Crear directorios necesarios
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Archivo HDF5 para almacenar todas las secuencias
        self.dataset_file = os.path.join(self.data_dir, 'sequences.h5')
        
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
            labels_map = {
                'sign_to_index': {},
                'index_to_sign': {},
                'num_classes': 0,
                'version': '2.1.0', # Version updated for HDF5
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
        """Cuenta secuencias ya recolectadas para una seña desde el archivo HDF5"""
        if not os.path.exists(self.dataset_file):
            return 0
            
        label_index = self.labels_map['sign_to_index'].get(sign)
        if label_index is None:
            return 0
            
        count = 0
        try:
            with h5py.File(self.dataset_file, 'r') as hf:
                if 'y' in hf:
                    y_data = hf['y'][:]
                    count = np.sum(y_data == label_index)
        except Exception as e:
            print(f"Error al leer el contador de secuencias de HDF5: {e}")
            return 0
        return int(count)

    def get_next_sequence_id(self, sign):
        """Obtiene el siguiente ID de secuencia para una seña"""
        return self.get_collected_sequences_count(sign) + 1
    
    def save_sequence(self, sequence_data, sign, sequence_id, metadata):
        """Guarda una secuencia en el dataset HDF5"""
        label_index = self.add_sign_to_labels(sign)
        
        # Asegurar que sequence_data tenga la forma correcta (sequence_length, features)
        if len(sequence_data.shape) == 3 and sequence_data.shape[0] == 1:
            sequence_data = sequence_data.reshape(sequence_data.shape[1], sequence_data.shape[2])
        
        try:
            with h5py.File(self.dataset_file, 'a') as hf:
                # Crear datasets si no existen
                if 'X' not in hf:
                    hf.create_dataset('X', data=[sequence_data], 
                                      maxshape=(None, sequence_data.shape[0], sequence_data.shape[1]), 
                                      chunks=True, dtype='float32')
                    hf.create_dataset('y', data=[label_index], 
                                      maxshape=(None,), 
                                      chunks=True, dtype='int32')
                else:
                    # Añadir nuevos datos
                    hf['X'].resize((hf['X'].shape[0] + 1), axis=0)
                    hf['X'][-1] = sequence_data
                    
                    hf['y'].resize((hf['y'].shape[0] + 1), axis=0)
                    hf['y'][-1] = label_index
        except Exception as e:
            print(f"Error al guardar la secuencia en HDF5: {e}")
            # Aquí se podría implementar una lógica de recuperación o limpieza
            return None, None

        # Guardar metadatos individuales (esto se mantiene igual)
        metadata_file = os.path.join(self.metadata_dir, f"{sign}_{sequence_id}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        # Actualizar info del dataset
        self._update_dataset_info(sign, sequence_id, metadata)
        
        return self.dataset_file, metadata_file

    def _update_dataset_info(self, sign, sequence_id, metadata):
        """Actualiza la información general del dataset"""
        dataset_info = self._load_dataset_info()
        
        dataset_info['total_sequences'] = dataset_info.get('total_sequences', 0) + 1
        
        if 'signs' not in dataset_info:
            dataset_info['signs'] = {}
        if sign not in dataset_info['signs']:
            dataset_info['signs'][sign] = {'count': 0, 'last_updated': None}
        
        dataset_info['signs'][sign]['count'] += 1
        dataset_info['signs'][sign]['last_updated'] = datetime.now().isoformat()
        dataset_info['last_updated'] = datetime.now().isoformat()
        
        with open(self.dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    def _load_dataset_info(self):
        """Carga la información del dataset"""
        if os.path.exists(self.dataset_info_file):
            with open(self.dataset_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                'version': '2.1.0',
                'created': datetime.now().isoformat(),
                'total_sequences': 0,
                'signs': {},
                'format': 'hdf5' # Formato actualizado
            }
            
    def create_metadata(self, sign, sign_type, hands_info, quality_score, 
                       quality_level, motion_features, issues, collection_mode="NORMAL"):
        """Crea estructura de metadatos para una secuencia"""
        # Esta función no necesita cambios, es independiente del formato de almacenamiento
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
            'version': '2.1.0'
        }
        return metadata

    def load_keras_dataset(self, signs: List[str] = None):
        """Carga el dataset completo o un subconjunto desde el archivo HDF5"""
        if not os.path.exists(self.dataset_file):
            return np.array([]), np.array([])

        with h5py.File(self.dataset_file, 'r') as hf:
            if 'X' not in hf or 'y' not in hf:
                return np.array([]), np.array([])

            if signs is None:
                # Cargar todo el dataset
                X_all = hf['X'][:]
                y_all = hf['y'][:]
                return X_all, y_all
            else:
                # Cargar solo las señas especificadas
                label_indices = [self.labels_map['sign_to_index'][s] for s in signs if s in self.labels_map['sign_to_index']]
                if not label_indices:
                    return np.array([]), np.array([])
                
                y_all = hf['y'][:]
                # Encontrar los índices de las filas que corresponden a las señas deseadas
                mask = np.isin(y_all, label_indices)
                
                X_filtered = hf['X'][mask]
                y_filtered = y_all[mask]
                
                return X_filtered, y_filtered

    def get_keras_dataset_info(self):
        """Obtiene información del dataset desde el archivo HDF5"""
        info = {
            'num_classes': self.labels_map['num_classes'],
            'labels_map': self.labels_map,
            'signs_available': [],
            'total_sequences': 0,
            'sequences_per_sign': {}
        }
        
        if not os.path.exists(self.dataset_file):
            return info

        try:
            with h5py.File(self.dataset_file, 'r') as hf:
                if 'y' in hf:
                    y_data = hf['y'][:]
                    info['total_sequences'] = len(y_data)
                    
                    unique_labels, counts = np.unique(y_data, return_counts=True)
                    
                    for label, count in zip(unique_labels, counts):
                        sign = self.labels_map['index_to_sign'].get(str(label))
                        if sign:
                            info['signs_available'].append(sign)
                            info['sequences_per_sign'][sign] = int(count)
        except Exception as e:
            print(f"Error al leer la información del dataset HDF5: {e}")

        return info

    def load_sequence(self, sign, sequence_id):
        """Carga una secuencia específica del dataset HDF5."""
        label_index = self.labels_map['sign_to_index'].get(sign)
        if label_index is None:
            return None, None

        if not os.path.exists(self.dataset_file):
            return None, None

        try:
            with h5py.File(self.dataset_file, 'r') as hf:
                if 'X' not in hf or 'y' not in hf:
                    return None, None
                
                y_data = hf['y'][:]
                # Encontrar los índices de todas las secuencias para esta seña
                sign_indices = np.where(y_data == label_index)[0]
                
                if sequence_id <= len(sign_indices):
                    # Obtener el índice real en el dataset HDF5
                    actual_index = sign_indices[sequence_id - 1]
                    sequence_data = hf['X'][actual_index]
                    
                    # Cargar metadatos
                    metadata_file = os.path.join(self.metadata_dir, f"{sign}_{sequence_id}_metadata.json")
                    metadata = None
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    
                    return sequence_data, metadata
        except Exception as e:
            print(f"Error cargando secuencia {sign}_{sequence_id} desde HDF5: {e}")
        
        return None, None

    def get_collection_statistics(self):
        """Obtiene estadísticas de la colección de datos (formato HDF5)"""
        stats = {
            'total_signs': self.labels_map.get('num_classes', 0),
            'total_sequences': 0,
            'signs_by_type': {},
            'quality_distribution': {'EXCELENTE': 0, 'BUENA': 0, 'ACEPTABLE': 0, 'REGULAR': 0, 'MALA': 0},
            'completion_status': {},
            'hdf5_format': True
        }
        
        dataset_info = self._load_dataset_info()
        stats['total_sequences'] = dataset_info.get('total_sequences', 0)
        stats['completion_status'] = {s: c['count'] for s, c in dataset_info.get('signs', {}).items()}

        # Para estadísticas más detalladas, se necesita iterar sobre los metadatos
        # Esta parte es más lenta pero necesaria para quality, etc.
        if os.path.exists(self.metadata_dir):
            for metadata_file in os.listdir(self.metadata_dir):
                if metadata_file.endswith('_metadata.json'):
                    try:
                        with open(os.path.join(self.metadata_dir, metadata_file), 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        sign_type = metadata.get('sign_type', 'unknown')
                        stats['signs_by_type'][sign_type] = stats['signs_by_type'].get(sign_type, 0) + 1
                        
                        quality_level = metadata.get('quality_level', 'UNKNOWN')
                        if quality_level in stats['quality_distribution']:
                            stats['quality_distribution'][quality_level] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        return stats

    def validate_keras_dataset_integrity(self):
        """Valida la integridad del dataset HDF5"""
        issues = []
        if not os.path.exists(self.dataset_file):
            issues.append(f"Archivo HDF5 no encontrado en: {self.dataset_file}")
            return issues

        try:
            with h5py.File(self.dataset_file, 'r') as hf:
                if 'X' not in hf:
                    issues.append("Dataset 'X' no encontrado en el archivo HDF5.")
                if 'y' not in hf:
                    issues.append("Dataset 'y' no encontrado en el archivo HDF5.")
                
                if 'X' in hf and 'y' in hf:
                    if hf['X'].shape[0] != hf['y'].shape[0]:
                        issues.append(f"Inconsistencia de datos: X tiene {hf['X'].shape[0]} secuencias, y tiene {hf['y'].shape[0]} etiquetas.")
        except Exception as e:
            issues.append(f"No se pudo abrir o leer el archivo HDF5: {e}")
            
        return issues

    def cleanup_keras_dataset(self):
        """Informa sobre la limpieza de datasets HDF5."""
        print("La limpieza de archivos HDF5 no se realiza in-situ.")
        print("Para eliminar datos corruptos o no deseados, se recomienda")
        print("crear un script que lea los datos válidos y los escriba en un nuevo archivo HDF5.")
        return 0

    def export_dataset_summary(self, output_file='dataset_summary.json'):
        """Exporta un resumen completo del dataset en formato HDF5"""
        stats = self.get_collection_statistics()
        keras_info = self.get_keras_dataset_info()
        
        summary = {
            'dataset_info': {
                'name': 'LSP Dataset - HDF5 Compatible',
                'version': '2.1.0',
                'format': 'hdf5',
                'creation_date': datetime.now().isoformat(),
                'data_directory': self.data_dir,
                'dataset_file': self.dataset_file,
                'metadata_directory': self.metadata_dir
            },
            'statistics': stats,
            'keras_info': keras_info,
        }
        
        output_path = os.path.join(self.metadata_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        return output_path

    def _get_keras_file_structure(self):
        """Obtiene la estructura de archivos del dataset en formato HDF5"""
        # Esta función ya no es tan relevante, pero se puede adaptar
        # para mostrar la estructura interna del HDF5 si es necesario.
        # Por ahora, la mantengo simple.
        structure = {
            'dataset_file': self.dataset_file,
            'metadata_files': [],
            'labels_map': self.labels_file
        }
        if os.path.exists(self.metadata_dir):
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('_metadata.json')]
            structure['metadata_files'] = metadata_files
        
        return structure