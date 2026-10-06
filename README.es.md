# 🚀 Sistema LSP - Lenguaje de Señas Peruano v2.0

> Sistema modular avanzado para recolección, entrenamiento e inferencia de señas del Lenguaje de Señas Peruano (LSP) usando GRU Bidireccional y MediaPipe

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11%2B-green.svg)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Tabla de Contenidos

- [🎯 Características Principales](#-características-principales)
- [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [🚀 Instalación y Configuración](#-instalación-y-configuración)
- [📊 Módulos del Sistema](#-módulos-del-sistema)
- [🔄 Data Augmentation](#-data-augmentation)
- [💡 Uso del Sistema](#-uso-del-sistema)
- [📈 Estadísticas y Progreso](#-estadísticas-y-progreso)
- [🧪 Testing](#-testing)
- [📚 Documentación Técnica](#-documentación-técnica)

## 🎯 Características Principales

### ✨ **Nuevas Funcionalidades v2.0**
- 🔄 **Data Augmentation Inteligente**: Reduce trabajo manual hasta 70%
- 📊 **Dashboard de Progreso**: Indicadores visuales de completación del dataset
- 🤖 **Descarga Automática de Modelos**: Setup automático de MediaPipe
- 📈 **Estadísticas Avanzadas**: Análisis detallado del dataset en tiempo real
- 🎯 **Formato Keras Optimizado**: Compatible con TensorFlow/Keras nativo

### 🧠 **Optimizaciones para GRU**
- **Secuencias de 60 frames**: Contexto temporal óptimo para GRU bidireccional
- **157 features por frame**: Landmarks de manos (126) + pose (24) + velocidad (7)
- **Normalización específica**: Preprocessing optimizado para redes recurrentes
- **Análisis temporal**: 20 métricas de movimiento para mejor entrenamiento

### 🏗️ **Arquitectura Modular**
- **Separación de responsabilidades**: 7 módulos especializados
- **Escalabilidad**: Fácil extensión y mantenimiento
- **Testing granular**: Verificación independiente por módulo
- **Reutilización**: Componentes intercambiables

## 🏗️ Arquitectura del Sistema

```
LSP_Final/
├── 🚀 run.py                     # Sistema principal con verificación automática
├── 📊 src/                       # Código fuente modular
│   ├── data_collection/          # Módulo de recolección
│   │   ├── main_collector.py     # ✅ Coordinador principal
│   │   ├── mediapipe_manager.py  # ✅ Gestión de MediaPipe
│   │   ├── feature_extractor.py  # ✅ Extracción de características
│   │   ├── motion_analyzer.py    # ✅ Análisis de movimiento
│   │   ├── ui_manager.py         # ✅ Interfaz de usuario mejorada
│   │   ├── data_manager.py       # ✅ Gestión de datos Keras
│   │   ├── sign_config.py        # ✅ Configuración de señas
│   │   └── data_augmentation.py  # 🆕 Data Augmentation
│   ├── training/                 # Módulo de entrenamiento
│   ├── evaluation/               # Módulo de evaluación
│   ├── inference/                # Módulo de inferencia
│   └── utils/                    # 🆕 Utilidades del sistema
│       └── mediapipe_model_downloader.py  # 🆕 Descarga automática
├── 📁 data/                      # Datos recolectados (formato Keras)
├── 🤖 models/                    # Modelos MediaPipe y entrenados
├── 🧪 tests/                     # Suite de pruebas completa
├── 📚 docs/                      # Documentación
└── 📋 requirements.txt           # Dependencias actualizadas
```

## 🚀 Instalación y Configuración

### 📋 Prerrequisitos
- Python 3.11 o superior
- Cámara web para recolección de datos
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio en disco

### ⚡ Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU.git
cd LSP_Final

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\\Scripts\\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar sistema (configuración automática)
python run.py
```

### 🤖 Configuración Automática

El sistema incluye **configuración automática** que:
- ✅ Verifica todas las dependencias
- ✅ Descarga modelos MediaPipe automáticamente
- ✅ Crea directorios necesarios
- ✅ Valida configuración del sistema

## 📊 Módulos del Sistema

### 1. 📊 **Data Collection** (Recolección de Datos)

#### Características Principales:
- **Menú mejorado con progreso**: Visualización en tiempo real del estado del dataset
- **Data Augmentation integrado**: Opción [A] para amplificar dataset automáticamente
- **Estadísticas detalladas**: Opción [S] para análisis completo
- **Formato Keras nativo**: Datos listos para TensorFlow/Keras

#### Opciones Disponibles:
```
[1-n] - Recolectar seña específica
[ALL] - Recolectar todas las señas
[A]   - Data Augmentation automático  🆕
[S]   - Ver estadísticas detalladas   🆕
[Q]   - Salir
```

### 2. 🔄 **Data Augmentation**

#### Técnicas Implementadas:
- **Variaciones temporales**: Cambios de velocidad, pausas, interpolación
- **Transformaciones espaciales**: Rotación, escala, traslación conservadora
- **Ruido controlado**: Gaussiano y jitter en landmarks
- **Variaciones de manos**: Intercambio izquierda/derecha

#### Modos de Augmentación:
- **Conservadora (50% reducción)**: Aumenta dataset manteniendo alta calidad
- **Moderada (70% reducción)**: Mayor expansión con calidad aceptable
- **Específica por seña**: Control granular por seña individual
- **Análisis detallado**: Reporte de potencial de augmentación

### 3. 🧠 **Training** (Entrenamiento)
- Arquitectura GRU bidireccional optimizada
- Carga automática de datasets Keras
- Hyperparameter tuning avanzado

### 4. 📈 **Evaluation** (Evaluación)
- Métricas especializadas para señas
- Validación cruzada temporal
- Análisis de confusión por categorías

### 5. 🎯 **Inference** (Inferencia)
- Traducción en tiempo real
- Confianza por predicción
- Pipeline optimizado

## 💡 Uso del Sistema

### 🚀 Inicio Rápido

```bash
# Ejecutar sistema principal
python run.py
```

Al ejecutar, el sistema automáticamente:
1. 🔍 Verifica dependencias
2. 📥 Descarga modelos MediaPipe si faltan
3. ✅ Confirma configuración
4. 🚀 Inicia menú principal

### 📊 Recolección de Datos

```bash
# Desde el menú principal, seleccionar opción 1
# Luego seguir el menú interactivo:

🚀 RECOLECTOR DE DATOS LSP - VERSIÓN MODULAR
═══════════════════════════════════════════════

📊 PROGRESO GENERAL DEL DATASET:
   📈 Progreso total: 47/190 secuencias (24.7%)
   ✅ Señas completadas: 0/5
   ⚠️ Secuencias faltantes: 143
   📊 [█████████░░░░░░░░░░░░░░░░░░░] 24.7%

📋 Señas disponibles:
   ⚠️  1. A     [3/30] (faltan 27)
   ⚠️  2. HOLA  [24/50] (faltan 26)
   ...
```

### 🔄 Data Augmentation

```python
# Ejemplo de uso programático
from src.data_collection.data_augmentation import LSPDataAugmenter

augmenter = LSPDataAugmenter()
augmented_sequences = augmenter.augment_sequence(
    sequence_data, 'word', metadata, num_augmentations=3
)
```

## 📈 Estadísticas y Progreso

### 📊 Dashboard de Progreso

El sistema incluye un dashboard completo que muestra:
- **Progreso general**: Porcentaje de completación del dataset
- **Estado por seña**: Secuencias recolectadas vs requeridas
- **Distribución por categoría**: Letras, palabras, frases
- **Calidad del dataset**: Distribución de calidad por nivel
- **Tiempo estimado**: Cálculo de trabajo restante

### 📋 Ejemplo de Reporte

```
📊 ESTADÍSTICAS DETALLADAS DEL DATASET LSP
═══════════════════════════════════════════

📈 RESUMEN GENERAL:
   🎯 Total de señas únicas: 25
   📝 Total de secuencias recolectadas: 340
   ✅ Señas completadas: 15/25 (60.0%)
   📊 Progreso general: 340/1250 (27.2%)
   ⚠️ Secuencias faltantes: 910

� DISTRIBUCIÓN POR CATEGORÍAS:
   📂 Letras estáticas: 180/750 (24.0%)
   📂 Palabras básicas: 100/300 (33.3%)
   📂 Saludos: 60/200 (30.0%)

⭐ DISTRIBUCIÓN POR CALIDAD:
   • EXCELENTE: 102 secuencias (30.0%)
   • BUENA: 170 secuencias (50.0%)
   • ACEPTABLE: 68 secuencias (20.0%)

💡 RECOMENDACIONES:
   • Enfócate en completar: letras dinámicas
   • Tiempo estimado restante: 30h 20m
   • Considera usar Data Augmentation (reduce 70% el trabajo)
```

## 🧪 Testing

### 🔧 Suite de Pruebas Completa

```bash
# Test completo del sistema
python tests/test_data_collection_keras.py

# Test específico de Data Augmentation
python tests/test_data_augmentation.py

# Test del menú mejorado
python tests/test_improved_menu.py
```

### ✅ Cobertura de Tests
- **Data Collection**: Verificación completa del pipeline
- **Data Augmentation**: Validación de todas las técnicas
- **Formato Keras**: Compatibilidad con TensorFlow
- **UI Mejorada**: Funcionalidad del menú con progreso

## 📚 Documentación Técnica

### 🔄 Data Augmentation

#### Configuración de Técnicas

```python
augmentation_config = {
    'temporal_variations': {
        'speed_range': (0.8, 1.2),    # ±20% velocidad
        'pause_probability': 0.1,      # 10% pausas
        'interpolation_factor': 1.2    # Factor interpolación
    },
    'spatial_transformations': {
        'rotation_range': (-15, 15),   # ±15 grados
        'scale_range': (0.9, 1.1),     # ±10% escala
        'translation_range': (-0.05, 0.05)  # ±5% traslación
    },
    'noise_augmentation': {
        'gaussian_std': 0.01,          # Ruido gaussiano
        'landmark_jitter': 0.005,      # Jitter landmarks
        'dropout_probability': 0.02    # 2% dropout
    }
}
```

#### Técnicas Seguras por Tipo de Seña

| Tipo de Seña | Técnicas Permitidas | Justificación |
|---------------|-------------------|---------------|
| **Letras estáticas** | Espacial + Ruido + Manos | Preserva forma estática |
| **Letras dinámicas** | Temporal + Espacial + Ruido | Mantiene movimiento esencial |
| **Palabras** | Temporal medio + Espacial + Ruido | Balance semántica/variabilidad |
| **Frases** | Temporal ligero + Ruido | Preserva secuencia temporal |

### 📊 Formato de Datos Keras

#### Estructura de Archivos

```
data/
├── sequences_X.npy          # Features: (samples, 60, 157)
├── sequences_y.npy          # Labels: (samples,)
├── sequences_metadata.json  # Metadatos completos
└── dataset_summary.json     # Resumen del dataset
```

#### Formato de Features

```python
# Estructura de features por frame (157 total):
features = [
    # Mano derecha: 21 landmarks × 3 coords = 63
    hand_right_landmarks,  # [x1, y1, conf1, x2, y2, conf2, ...]
    
    # Mano izquierda: 21 landmarks × 3 coords = 63  
    hand_left_landmarks,   # [x1, y1, conf1, x2, y2, conf2, ...]
    
    # Pose: 8 landmarks × 3 coords = 24
    pose_landmarks,        # [x1, y1, conf1, x2, y2, conf2, ...]
    
    # Velocidades temporales: 7
    velocity_features      # [vel_hands, vel_pose, accel_hands, ...]
]
```

## 🔧 Configuración Avanzada

### 🎯 Personalización de Señas

```python
# Agregar nueva seña
sign_config = {
    'NUEVA_SEÑA': {
        'sign_type': 'word',
        'hands_required': 2,
        'is_dynamic': True,
        'instructions': 'Descripción de la seña...',
        'learning_tips': ['Tip 1', 'Tip 2'],
        'quality_thresholds': {
            'min_movement': 0.02,
            'max_noise': 0.05
        }
    }
}
```

### ⚡ Optimización de Rendimiento

```python
# Configuración para diferentes hardware
performance_config = {
    'low_end': {
        'sequence_length': 30,      # Frames reducidos
        'features_simplified': True, # Features básicas
        'quality_fast': True        # Evaluación rápida
    },
    'high_end': {
        'sequence_length': 60,      # Frames completos
        'features_advanced': True,  # Todas las features
        'quality_detailed': True    # Evaluación completa
    }
}
```

## 🤝 Contribución

### 📋 Guías de Desarrollo

1. **Fork del repositorio**
2. **Crear branch específica**: `git checkout -b feature/nueva-funcionalidad`
3. **Seguir estándares de código**: PEP 8, type hints, documentación
4. **Escribir tests**: Cobertura mínima 80%
5. **Actualizar documentación**: README y docstrings
6. **Pull request**: Descripción detallada de cambios

### � Testing Guidelines

```bash
# Ejecutar todos los tests antes de commit
python -m pytest tests/ -v

# Verificar cobertura
python -m pytest tests/ --cov=src --cov-report=html

# Tests específicos por módulo
python -m pytest tests/test_data_augmentation.py -v
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **MediaPipe Team**: Por la excelente biblioteca de landmarks
- **TensorFlow Team**: Por el framework de ML
- **Comunidad LSP**: Por el apoyo y validación de señas
- **Colaboradores**: Ver [CONTRIBUTORS.md](CONTRIBUTORS.md)

## 📞 Soporte

- **Issues**: [GitHub Issues](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/issues)
- **Documentación**: [Wiki del proyecto](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/wiki)
- **Email**: [Contacto del proyecto]

---

<div align="center">

**🚀 Sistema LSP v2.0 - Democratizando el acceso al Lenguaje de Señas Peruano** 

[![GitHub stars](https://img.shields.io/github/stars/Jaed69/Traductor-Lenguaje-Senas-GRU.svg?style=social&label=Star)](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU)
[![GitHub forks](https://img.shields.io/github/forks/Jaed69/Traductor-Lenguaje-Senas-GRU.svg?style=social&label=Fork)](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/fork)

</div>
