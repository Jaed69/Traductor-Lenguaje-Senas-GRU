# Guía de Desarrollo

Esta guía te ayudará a configurar el entorno de desarrollo para el Traductor de Lenguaje de Señas.

## Pre-requisitos

- Python 3.8 o superior
- Una cámara web funcional
- Git

## Configuración del Entorno

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### 2. Crear Entorno Virtual

**Con conda (recomendado):**
```bash
conda create -n traductor_senas python=3.9
conda activate traductor_senas
```

**Con venv:**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 3. Instalar Dependencias

**Opción 1: Dependencias principales (recomendado para desarrollo)**
```bash
pip install -r requirements_clean.txt
```

**Opción 2: Dependencias exactas (para reproducir el entorno exacto)**
```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación

```bash
python -c "import cv2, mediapipe, tensorflow; print('¡Todas las dependencias instaladas correctamente!')"
```

## Flujo de Desarrollo

### 1. Recolección de Datos
```bash
python data_collector.py
```

### 2. Entrenamiento del Modelo
```bash
python model_trainer_sequence.py
```

### 3. Prueba del Traductor
```bash
python main.py
```

## Estructura de Archivos Generados

Después de ejecutar los scripts, tendrás:

```
data/
├── sequences/              # Datos de entrenamiento
│   ├── HOLA/              # Secuencias para "HOLA"
│   ├── A/                 # Secuencias para "A"
│   └── ...
├── sign_model_gru.h5      # Modelo entrenado
└── label_encoder.npy      # Codificador de etiquetas
```

## Consejos de Desarrollo

1. **Recolección de Datos**: Asegúrate de tener buena iluminación y un fondo uniforme
2. **Entrenamiento**: El primer entrenamiento puede tomar varios minutos
3. **Predicción**: La precisión mejora con más datos de entrenamiento

## Solución de Problemas

### Error de Cámara
- Verifica que tu cámara web esté conectada y funcionando
- Cierra otras aplicaciones que puedan estar usando la cámara

### Error de ImportError
- Asegúrate de que todas las dependencias estén instaladas
- Verifica que el entorno virtual esté activado

### Modelo no encontrado
- Ejecuta primero `data_collector.py` y luego `model_trainer_sequence.py`
- Verifica que los archivos se hayan generado en la carpeta `data/`

## Recursos Adicionales

- [Documentación de MediaPipe](https://mediapipe.dev/)
- [Documentación de TensorFlow](https://www.tensorflow.org/)
- [Guía de OpenCV](https://docs.opencv.org/)
