# 🚀 Guía de Instalación y Configuración LSP v2.0

## 📋 Índice
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación Paso a Paso](#instalación-paso-a-paso)
- [Configuración Automática](#configuración-automática)
- [Verificación de Instalación](#verificación-de-instalación)
- [Configuración Manual](#configuración-manual)
- [Solución de Problemas](#solución-de-problemas)

## Requisitos del Sistema

### 🖥️ Hardware Mínimo
- **CPU**: Intel i5 / AMD Ryzen 5 (4 núcleos)
- **RAM**: 4GB (8GB recomendado)
- **Almacenamiento**: 2GB libres
- **Cámara**: Webcam 720p mínimo (1080p recomendado)
- **GPU**: Opcional (CUDA compatible para aceleración)

### 💻 Software Requerido
- **SO**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.11 o superior
- **Git**: Para clonar repositorio
- **Navegador**: Para visualización de resultados

### 📦 Dependencias Principales
```
tensorflow>=2.13.0
mediapipe>=0.10.11
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
requests>=2.31.0
tqdm>=4.65.0
```

## Instalación Paso a Paso

### 🚀 Método 1: Instalación Automática (Recomendado)

#### 1. Clonar Repositorio
```bash
# Clonar desde GitHub
git clone https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU.git
cd LSP_Final
```

#### 2. Crear Entorno Virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Activar entorno (Linux/macOS)
source venv/bin/activate
```

#### 3. Instalación Automática
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar sistema con configuración automática
python run.py
```

**¡Listo!** El sistema se configurará automáticamente:
- ✅ Verificará dependencias
- ✅ Descargará modelos MediaPipe
- ✅ Creará directorios necesarios
- ✅ Validará configuración

### 🛠️ Método 2: Instalación Manual

#### 1. Instalación de Python 3.11+
```bash
# Verificar versión de Python
python --version

# Si es menor a 3.11, instalar desde python.org
# O usar pyenv (Linux/macOS)
pyenv install 3.11.7
pyenv global 3.11.7
```

#### 2. Instalación de Dependencias Core
```bash
# Actualizar pip
pip install --upgrade pip

# Instalar TensorFlow
pip install tensorflow>=2.13.0

# Instalar MediaPipe
pip install mediapipe>=0.10.11

# Instalar OpenCV
pip install opencv-python>=4.8.0
```

#### 3. Dependencias Científicas
```bash
# NumPy y SciPy
pip install numpy>=1.24.0 scipy>=1.10.0

# Scikit-learn
pip install scikit-learn>=1.3.0

# Matplotlib para visualización
pip install matplotlib>=3.7.0
```

#### 4. Utilidades Adicionales
```bash
# Requests para descargas
pip install requests>=2.31.0

# TQDM para barras de progreso
pip install tqdm>=4.65.0

# Pillow para procesamiento de imágenes
pip install Pillow>=10.0.0
```

## Configuración Automática

### 🤖 Sistema de Auto-Setup

El sistema LSP incluye un **motor de configuración automática** que:

#### 1. Verificación de Dependencias
```python
def verify_dependencies():
    """Verifica todas las dependencias del sistema"""
    checks = {
        'tensorflow': check_tensorflow(),
        'mediapipe': check_mediapipe(), 
        'opencv': check_opencv(),
        'numpy': check_numpy(),
        'sklearn': check_sklearn()
    }
    return all(checks.values())
```

#### 2. Descarga Automática de Modelos
```python
def setup_mediapipe_models():
    """Descarga modelos MediaPipe automáticamente"""
    models = [
        'hand_landmarker.task',
        'pose_landmarker_heavy.task'
    ]
    
    for model in models:
        if not model_exists(model):
            download_model(model)
            verify_model(model)
```

#### 3. Creación de Estructura de Directorios
```python
def create_directory_structure():
    """Crea estructura completa de directorios"""
    directories = [
        'data/sequences',
        'models',
        'logs',
        'exports',
        'temp'
    ]
    
    for directory in directories:
        create_if_not_exists(directory)
```

### ⚙️ Configuración Personalizada

#### Variables de Entorno (Opcional)
```bash
# Configuración de GPU (si disponible)
export CUDA_VISIBLE_DEVICES=0

# Configuración de memoria TensorFlow
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Configuración de logging
export LSP_LOG_LEVEL=INFO
```

#### Archivo de Configuración (config.yaml)
```yaml
# config.yaml (se crea automáticamente)
system:
  data_path: "./data"
  models_path: "./models"
  log_level: "INFO"
  
mediapipe:
  confidence_threshold: 0.7
  tracking_confidence: 0.5
  
data_collection:
  sequence_length: 60
  fps: 30
  quality_threshold: 0.8
  
augmentation:
  enabled: true
  default_factor: 3
  quality_threshold: 0.85
```

## Verificación de Instalación

### ✅ Tests de Verificación

#### 1. Verificación Rápida
```bash
# Test básico del sistema
python -c "from src.data_collection.main_collector import LSPDataCollector; print('✅ Sistema instalado correctamente')"
```

#### 2. Test de MediaPipe
```bash
# Test de modelos MediaPipe
python -c "
from src.data_collection.mediapipe_manager import MediaPipeManager
mp = MediaPipeManager()
print('✅ MediaPipe funcionando correctamente')
"
```

#### 3. Test de Data Augmentation
```bash
# Test de Data Augmentation
python quick_test_augmentation.py
```

#### 4. Test Completo del Sistema
```bash
# Suite completa de tests
python tests/test_system_integration.py
```

### 📊 Dashboard de Verificación

Al ejecutar `python run.py`, verás:

```
🚀 SISTEMA LSP v2.0 - VERIFICACIÓN DE INSTALACIÓN
═══════════════════════════════════════════════════

✅ Python 3.11.7 - Compatible
✅ TensorFlow 2.13.0 - Compatible  
✅ MediaPipe 0.10.11 - Compatible
✅ OpenCV 4.8.1 - Compatible
✅ NumPy 1.24.3 - Compatible

🤖 MODELOS MEDIAPIPE:
✅ hand_landmarker.task - Disponible
✅ pose_landmarker_heavy.task - Disponible

📁 ESTRUCTURA DE DIRECTORIOS:
✅ data/ - Creado
✅ models/ - Creado
✅ logs/ - Creado

🎯 SISTEMA LISTO PARA USAR
```

## Configuración Manual

### 🔧 Configuración Avanzada

#### 1. Configuración de GPU (CUDA)
```bash
# Verificar disponibilidad de GPU
python -c "
import tensorflow as tf
print('GPUs disponibles:', len(tf.config.experimental.list_physical_devices('GPU')))
print('CUDA disponible:', tf.test.is_built_with_cuda())
"

# Configurar uso de memoria GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### 2. Optimización de Performance
```python
# config_performance.py
import tensorflow as tf

# Configuración para CPU optimizado
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Configuración para GPU (si disponible)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### 3. Configuración de Logging
```python
# config_logging.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lsp_system.log'),
        logging.StreamHandler()
    ]
)
```

### 📝 Configuración de Desarrollo

#### 1. Instalación de Herramientas de Desarrollo
```bash
# Herramientas de desarrollo
pip install pytest pytest-cov black flake8 mypy

# Jupyter para análisis
pip install jupyter notebook ipywidgets
```

#### 2. Pre-commit Hooks
```bash
# Instalar pre-commit
pip install pre-commit

# Configurar hooks
pre-commit install
```

#### 3. Configuración de IDE
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

## Solución de Problemas

### ❌ Problemas Comunes

#### 1. **Error de Versión de Python**
```
Error: Python 3.11+ requerido, encontrado 3.9.x
```

**Solución:**
```bash
# Instalar Python 3.11 desde python.org
# O usar pyenv (Linux/macOS)
pyenv install 3.11.7
pyenv local 3.11.7

# Recrear entorno virtual
rm -rf venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### 2. **Error de TensorFlow**
```
Error: Could not find a version that satisfies the requirement tensorflow>=2.13.0
```

**Solución:**
```bash
# Actualizar pip
pip install --upgrade pip

# Instalar versión específica
pip install tensorflow==2.13.0

# Si falla, usar versión CPU
pip install tensorflow-cpu==2.13.0
```

#### 3. **Error de MediaPipe**
```
Error: No module named 'mediapipe'
```

**Solución:**
```bash
# Verificar arquitectura del sistema
python -c "import platform; print(platform.machine())"

# Instalar versión específica
pip install mediapipe==0.10.11

# Si falla en Mac M1/M2
pip install mediapipe-silicon==0.10.11
```

#### 4. **Error de OpenCV**
```
Error: ImportError: No module named 'cv2'
```

**Solución:**
```bash
# Desinstalar versiones conflictivas
pip uninstall opencv-python opencv-contrib-python opencv-python-headless

# Reinstalar versión correcta
pip install opencv-python==4.8.1.78
```

#### 5. **Error de Cámara**
```
Error: Camera not accessible
```

**Solución:**
```bash
# Verificar permisos de cámara (macOS/Linux)
sudo chmod 666 /dev/video0

# Windows: Verificar configuración de privacidad
# Configuración > Privacidad > Cámara > Permitir aplicaciones
```

### 🔧 Herramientas de Diagnóstico

#### 1. Script de Diagnóstico Completo
```python
# diagnosis.py
import sys
import platform
import subprocess

def run_diagnosis():
    print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA LSP")
    print("=" * 50)
    
    # Información del sistema
    print(f"SO: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Verificar dependencias
    dependencies = [
        'tensorflow', 'mediapipe', 'opencv-python', 
        'numpy', 'sklearn', 'matplotlib'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep}: Instalado")
        except ImportError:
            print(f"❌ {dep}: NO instalado")
    
    # Verificar cámara
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Cámara: Accesible")
            cap.release()
        else:
            print("❌ Cámara: No accesible")
    except:
        print("❌ Cámara: Error al verificar")

if __name__ == "__main__":
    run_diagnosis()
```

#### 2. Test de Performance
```python
# performance_test.py
import time
import numpy as np
import tensorflow as tf

def performance_test():
    print("⚡ TEST DE PERFORMANCE")
    print("=" * 30)
    
    # Test CPU
    start = time.time()
    x = np.random.random((1000, 1000))
    np.dot(x, x)
    cpu_time = time.time() - start
    print(f"CPU Performance: {cpu_time:.3f}s")
    
    # Test GPU (si disponible)
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            start = time.time()
            x = tf.random.normal((1000, 1000))
            tf.linalg.matmul(x, x)
            gpu_time = time.time() - start
            print(f"GPU Performance: {gpu_time:.3f}s")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU: No disponible")

if __name__ == "__main__":
    performance_test()
```

### 📞 Obtener Ayuda

#### 1. Logs del Sistema
```bash
# Ver logs recientes
tail -f logs/lsp_system.log

# Buscar errores específicos
grep "ERROR" logs/lsp_system.log
```

#### 2. Información del Sistema
```bash
# Generar reporte completo
python diagnosis.py > system_report.txt

# Compartir reporte para soporte
```

#### 3. Contacto y Soporte
- **Issues**: [GitHub Issues](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/issues)
- **Documentación**: [Wiki del proyecto](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/wiki)
- **Email**: [Soporte técnico]

---

## 🎯 Checklist de Instalación

- [ ] Python 3.11+ instalado
- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas desde requirements.txt
- [ ] Sistema ejecutado con `python run.py`
- [ ] Verificación automática completada
- [ ] Modelos MediaPipe descargados
- [ ] Cámara funcionando correctamente
- [ ] Tests básicos pasados
- [ ] Documentación revisada

**🚀 ¡Sistema LSP v2.0 listo para usar!**

---

*Para más información detallada, consulta la [documentación completa](../README.md) y las [guías específicas](../docs/).*
