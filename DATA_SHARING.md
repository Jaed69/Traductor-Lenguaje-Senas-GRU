# Sistema de Compartición de Datos

Este documento explica cómo compartir y colaborar con los datos de entrenamiento para escalar el dataset del traductor de lenguaje de señas.

## 🎯 Objetivo

Permitir que múltiples usuarios contribuyan con datos de entrenamiento para crear un dataset más grande y diverso, mejorando la precisión del modelo para diferentes personas y estilos de señas.

## 📊 Estructura de Datos Compartidos

### Formato de Intercambio
Los datos se organizan en el siguiente formato para facilitar el intercambio:

```
shared_data/
├── metadata.json              # Información del dataset
├── contributors.json          # Lista de contribuidores
└── sequences/                 # Datos de secuencias
    ├── user_001/              # Datos del usuario 1
    │   ├── HOLA/
    │   ├── A/
    │   └── ...
    ├── user_002/              # Datos del usuario 2
    │   ├── HOLA/
    │   ├── A/
    │   └── ...
    └── merged/                # Dataset combinado
        ├── HOLA/
        ├── A/
        └── ...
```

## 🔄 Flujo de Contribución

### 1. Exportar tus datos
```bash
python data_exporter.py --output shared_data/user_[tu_id]/
```

### 2. Compartir datos
- Comprimir la carpeta de tu usuario
- Subir a Google Drive, Dropbox, o repositorio Git LFS
- Compartir el enlace con el equipo

### 3. Importar datos de otros usuarios
```bash
python data_importer.py --import shared_data/user_[otro_id]/
```

### 4. Entrenar con dataset combinado
```bash
python model_trainer_sequence.py --use-merged-data
```

## 📋 Metadatos del Dataset

Cada contribución incluye metadatos para garantizar la calidad:

```json
{
  "contributor_id": "user_001",
  "contributor_name": "Juan Pérez",
  "collection_date": "2025-06-29",
  "total_sequences": 1200,
  "signs_included": ["A", "B", "C", "HOLA", "GRACIAS"],
  "camera_specs": {
    "resolution": "1920x1080",
    "fps": 30
  },
  "lighting_conditions": "good",
  "background": "uniform",
  "quality_score": 8.5
}
```

## 🛠️ Herramientas de Colaboración

### Scripts Incluidos:
- `data_exporter.py` - Exporta tus datos en formato estándar
- `data_importer.py` - Importa datos de otros usuarios
- `data_merger.py` - Combina múltiples datasets
- `data_validator.py` - Valida la calidad de los datos
- `dataset_stats.py` - Estadísticas del dataset combinado

## 📈 Escalabilidad

### Beneficios del Dataset Colaborativo:
1. **Mayor Diversidad**: Diferentes personas, estilos de señas
2. **Mejor Generalización**: El modelo funciona mejor con usuarios nuevos
3. **Datos Balanceados**: Más ejemplos por cada seña
4. **Detección de Outliers**: Identificar datos de baja calidad

### Métricas de Calidad:
- Mínimo 40 secuencias por seña por usuario
- Consistencia en la velocidad de las señas
- Calidad de la detección de landmarks
- Diversidad en condiciones de grabación

## 🔐 Privacidad y Ética

### Consideraciones:
- Los datos solo contienen landmarks (puntos clave), no imágenes reales
- Cada contribuidor mantiene anonimato (user_001, user_002, etc.)
- Opción de contribuir anónimamente
- Derecho a retirar datos en cualquier momento

## 🚀 Implementación

### Fase 1: Scripts de Exportación/Importación
- Crear herramientas para intercambio de datos
- Validación automática de calidad

### Fase 2: Plataforma de Colaboración
- Sistema web para subir/descargar datos
- Dashboard con estadísticas del dataset

### Fase 3: Entrenamiento Distribuido
- Modelo que aprende incrementalmente
- Validación cruzada entre contribuidores
