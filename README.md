# 🚀 Traductor de Lenguaje de Señas Peruano (LSP) - Versión 2.0

## 🆕 **NUEVA VERSIÓN 2.0 - Recolector de Datos Optimizado para GRU**

### ✨ **Características Principales v2.0:**

- 🧠 **Optimizado para GRU Bidireccional** con keras.GRU
- 🏃‍♂️ **Tracking de hombros** para expresiones corporales (GRACIAS, POR FAVOR)
- 📊 **171 features avanzadas:** Manos (126) + Pose (36) + Velocidades (9)
- 🎯 **Secuencias de 60 frames** para mejor contexto temporal
- 📈 **24 métricas de calidad** incluyendo análisis de hombros
- 🔧 **API de MediaPipe Tasks** moderna (>=0.10.11)
- ⚡ **Procesamiento asíncrono** para máxima eficiencia

### 🎬 **Señas Soportadas:**

#### 📝 **Alfabeto LSP (Estáticas/Dinámicas):**
- **Estáticas (24):** A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
- **Dinámicas (5):** J, Z, Ñ, RR, LL

#### 🗣️ **Palabras y Expresiones:**
- **Básicas:** AMOR, CASA, FAMILIA, ESCUELA
- **Saludos:** HOLA, ADIÓS, BUENOS DÍAS, BUENAS NOCHES
- **Cortesía:** GRACIAS, POR FAVOR, MUCHO GUSTO, DE NADA
- **Conversación:** CÓMO ESTÁS

### 🛠️ **Instalación y Configuración:**

#### 1. **Requisitos del Sistema:**
```bash
Python 3.11+
Cámara web funcional
Windows/Linux/macOS
```

#### 2. **Instalar Dependencias:**
```bash
pip install -r requirements.txt
```

#### 3. **Descargar Modelos de MediaPipe:**
Crear carpeta `models/` y descargar:
- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [pose_landmarker_heavy.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task)

#### 4. **Ejecutar Recolector:**
```bash
python data_c.py
```

### 🎯 **Novedades Versión 2.0:**

#### 🏃‍♂️ **Tracking de Hombros y Pose Corporal:**
- **12 puntos de pose estratégicos** (cabeza, hombros, brazos, torso)
- **Análisis de simetría de hombros** para expresiones bilaterales
- **Coordinación mano-hombro** especial para señas como "GRACIAS"
- **Velocidades específicas** de hombros y torso superior

#### 🧠 **Optimización para GRU Bidireccional:**
- **Secuencias de 60 frames** vs 30 anteriores
- **Normalización específica** para rango [-1, 1] ideal para GRU
- **9 tipos de velocidades** para análisis temporal completo
- **24 métricas de calidad** incluyendo periodicidad y suavidad

#### 📊 **Sistema de Calidad Avanzado:**
- **Evaluación específica por tipo** de seña (estática/dinámica/expresiva)
- **Criterios especiales** para señas con componente corporal
- **Bonus por coordinación corporal** en expresiones
- **Detección automática** de problemas de calidad

#### 🎮 **Interfaz Mejorada:**
- **HUD informativo** con estado de tracking en tiempo real
- **Indicadores visuales** separados para manos y hombros
- **Progress bars** y métricas de calidad instantáneas
- **Modo automático** de mejora de calidad

### 📈 **Especificaciones Técnicas:**

| Característica | Versión 1.0 | **Versión 2.0** |
|---|---|---|
| Features totales | 126 | **171** |
| Métricas de calidad | 12 | **24** |
| Puntos de pose | 8 | **12** |
| Tipos de velocidad | 2 | **9** |
| Frames por secuencia | 30 | **60** |
| Soporte de expresiones | ❌ | **✅** |
| Tracking de hombros | ❌ | **✅** |
| API MediaPipe | Classic | **Tasks** |

### 🎬 **Uso del Sistema:**

#### **Menú Principal:**
1. **Recolectar seña específica** - Elegir seña individual
2. **Recolectar por categoría** - Alfabeto, palabras, expresiones
3. **Modo mejora de calidad** - Sustituir secuencias de baja calidad
4. **Estadísticas detalladas** - Progreso y análisis de calidad
5. **Salir** - Finalizar sesión

#### **Controles de Recolección:**
- `ESPACIO`: Iniciar/pausar grabación
- `Q`: Salir de recolección
- Barra de progreso visual durante grabación

#### **Calidad de Datos:**
- **EXCELENTE (92%+):** Óptimo para GRU + Expresiones
- **BUENA (80-91%):** Aceptable para GRU + Expresiones
- **REGULAR (65-79%):** Requiere mejora para GRU
- **MALA (<65%):** Inadecuada para GRU

### 🚀 **Para Desarrolladores:**

#### **Estructura de Datos:**
```python
# Formato de secuencia guardada (.npy)
sequence_data.shape = (60, 171)  # 60 frames × 171 features

# Features breakdown:
# [0:126]   - Hand landmarks (2 hands × 63 features)
# [126:162] - Pose landmarks (12 points × 3 coords)
# [162:171] - Velocities (9 different velocity types)
```

#### **Metadatos por Secuencia:**
```json
{
  "sign": "GRACIAS",
  "sign_type": "dynamic_two_hands",
  "quality_score": 95.2,
  "motion_features": [24 metrics],
  "shoulder_coordination": 0.87,
  "timestamp": "2025-07-10T..."
}
```

### 🔬 **Métricas de Hombros (Nuevas en v2.0):**

1. **Movimiento de hombros:** Actividad específica de hombros
2. **Simetría de hombros:** Balance entre hombro izquierdo/derecho
3. **Coordinación mano-hombro:** Sincronización temporal
4. **Amplitud de torso superior:** Rango de movimiento corporal

### 🎯 **Casos de Uso Especiales:**

#### **Señas Expresivas (Optimizadas):**
- **GRACIAS** 🙏: Movimiento característico hacia adelante
- **POR FAVOR** 🙏: Coordinación mano-pecho-hombro
- **MUCHO GUSTO** 🤝: Expresión corporal completa
- **BUENOS DÍAS** 🌅: Apertura corporal expresiva

#### **Para Entrenar GRU:**
```python
# Características ideales para keras.GRU:
- Secuencias de 60 frames (contexto temporal rico)
- 171 features normalizadas [-1, 1]
- 9 tipos de velocidades para análisis temporal
- Datos de alta calidad (>80% score)
```

### 📚 **Archivos del Proyecto:**

- `data_c.py` - Recolector principal v2.0
- `requirements.txt` - Dependencias actualizadas
- `models/` - Modelos de MediaPipe Tasks
- `data/sequences_advanced/` - Dataset recolectado

### 🤝 **Contribuciones:**

¡Las contribuciones son bienvenidas! Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

### 📄 **Licencia:**

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

### 👥 **Autores:**

- **Desarrollador Principal:** [Tu Nombre]
- **Versión 2.0:** Optimizada para GRU Bidireccional
- **Especialización:** Tracking de hombros y expresiones corporales

---

## 🎉 **¡Experimenta la nueva generación de recolección de datos LSP!**

### 📞 **Soporte:**
- Issues: [GitHub Issues](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/issues)
- Documentación: Este README.md
- Ejemplos: Carpeta `examples/` (próximamente)
