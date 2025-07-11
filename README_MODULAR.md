# Recolector de Datos LSP - Versión Modular 2.0

## Estructura del Proyecto

```
LSP_Final/
├── src/                           # Código fuente modular
│   ├── __init__.py               # Inicialización del paquete
│   ├── main_collector.py         # Clase principal coordinadora
│   ├── mediapipe_manager.py      # Gestión de MediaPipe
│   ├── feature_extractor.py      # Extracción de características
│   ├── motion_analyzer.py        # Análisis de movimiento y calidad
│   ├── ui_manager.py             # Interfaz de usuario y visualización
│   ├── data_manager.py           # Gestión de datos y almacenamiento
│   └── sign_config.py            # Configuración de señas
├── data/                         # Datos recolectados
│   └── sequences_advanced/       # Secuencias con metadatos
├── models/                       # Modelos de MediaPipe
│   ├── hand_landmarker.task
│   └── pose_landmarker_heavy.task
├── data_c.py                     # Versión original (legacy)
├── run_collector.py              # Punto de entrada principal
└── requirements.txt              # Dependencias
```

## Características Principales

### 🏗️ Arquitectura Modular
- **Separación de responsabilidades**: Cada módulo tiene una función específica
- **Mantenimiento fácil**: Cambios aislados en módulos específicos
- **Escalabilidad**: Fácil agregar nuevas funcionalidades
- **Reutilización**: Módulos reutilizables en otros proyectos

### 🧠 Optimizado para GRU
- Secuencias de 60 frames para mejor contexto temporal
- Normalización específica para redes recurrentes
- 20 métricas de movimiento avanzadas
- Features temporales para análisis secuencial

### 📊 Gestión Avanzada de Datos
- Metadatos completos por secuencia
- Validación de integridad del dataset
- Estadísticas automáticas de calidad
- Exportación de resúmenes del dataset

## Módulos Explicados

### 1. `mediapipe_manager.py`
**Responsabilidad**: Gestión de MediaPipe
- Inicialización de modelos
- Configuración de callbacks
- Manejo de resultados asíncronos
- Abstracción de la API de MediaPipe

### 2. `feature_extractor.py`
**Responsabilidad**: Extracción y procesamiento de características
- Normalización de landmarks
- Cálculo de velocidades temporales
- Optimización para GRU
- Normalización de features

### 3. `motion_analyzer.py`
**Responsabilidad**: Análisis de movimiento y calidad
- 20 métricas de movimiento avanzadas
- Evaluación de calidad específica por tipo de seña
- Detección de problemas comunes
- Análisis temporal para GRU

### 4. `ui_manager.py`
**Responsabilidad**: Interfaz de usuario y visualización
- Visualización de landmarks
- HUD informativo
- Menús interactivos
- Feedback en tiempo real

### 5. `data_manager.py`
**Responsabilidad**: Gestión de datos y almacenamiento
- Guardado de secuencias y metadatos
- Gestión de directorios
- Estadísticas del dataset
- Validación de integridad

### 6. `sign_config.py`
**Responsabilidad**: Configuración de señas
- Clasificación de tipos de señas
- Instrucciones específicas
- Configuración por tipo
- Consejos de aprendizaje

### 7. `main_collector.py`
**Responsabilidad**: Coordinación de todos los módulos
- Orquestación del flujo principal
- Coordinación entre módulos
- Lógica de recolección
- Manejo de estado global

## Ventajas de la Estructura Modular

### ✅ Mantenimiento
- **Aislamiento de cambios**: Modificar un módulo no afecta otros
- **Debugging fácil**: Problemas localizados por módulo
- **Testing granular**: Testear cada módulo independientemente

### ✅ Escalabilidad
- **Nuevas funcionalidades**: Agregar módulos sin modificar existentes
- **Diferentes backends**: Cambiar MediaPipe por otro framework
- **Múltiples UI**: Agregar interfaz web, móvil, etc.

### ✅ Reutilización
- **Otros proyectos**: Usar módulos en diferentes aplicaciones
- **Componentes intercambiables**: Swap de implementaciones
- **APIs consistentes**: Interfaces bien definidas

### ✅ Colaboración
- **Desarrollo paralelo**: Diferentes desarrolladores por módulo
- **Especialización**: Expertos en áreas específicas
- **Code reviews**: Revisiones focalizadas

## Uso del Sistema Modular

### Ejecución Principal
```bash
python run_collector.py
```

### Importación de Módulos
```python
from src.feature_extractor import FeatureExtractor
from src.motion_analyzer import MotionAnalyzer
# etc.
```

### Extensión del Sistema
```python
# Ejemplo: Nuevo analizador de calidad
from src.motion_analyzer import MotionAnalyzer

class AdvancedMotionAnalyzer(MotionAnalyzer):
    def custom_analysis(self):
        # Nueva funcionalidad
        pass
```

## Comparación con Versión Original

| Aspecto | Versión Original | Versión Modular |
|---------|------------------|-----------------|
| **Archivos** | 1 archivo (500+ líneas) | 7 módulos especializados |
| **Mantenimiento** | Difícil | Fácil |
| **Testing** | Complejo | Granular |
| **Extensibilidad** | Limitada | Alta |
| **Legibilidad** | Compleja | Clara |
| **Debugging** | Difícil | Fácil |

## Recomendaciones de Desarrollo

### 🔄 Flujo de Trabajo
1. **Modificaciones pequeñas**: Un módulo a la vez
2. **Testing**: Probar módulo antes de integrar
3. **Documentación**: Actualizar docs por módulo
4. **Versionado**: Semantic versioning por módulo

### 📋 Buenas Prácticas
- **Interfaces claras**: APIs bien definidas entre módulos
- **Error handling**: Manejo de errores por módulo
- **Logging**: Logs específicos por módulo
- **Configuration**: Configuración centralizada cuando sea necesario

Esta estructura modular te permitirá mantener y escalar tu proyecto de manera mucho más eficiente. ¿Te gustaría que profundice en algún módulo específico o que agregue alguna funcionalidad particular?
