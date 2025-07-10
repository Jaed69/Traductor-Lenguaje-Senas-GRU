# 📋 CHANGELOG - Traductor LSP

## [2.0.0] - 2025-07-10

### 🚀 **NUEVA VERSIÓN MAYOR - Recolector Optimizado para GRU**

#### ✨ **Características Nuevas:**
- **Tracking de hombros y pose corporal** con 12 puntos estratégicos
- **171 features avanzadas** (vs 126 anteriores): Manos + Pose + Velocidades
- **24 métricas de calidad** incluyendo análisis específico de hombros
- **Secuencias de 60 frames** para mejor contexto temporal GRU
- **9 tipos de velocidades** para análisis temporal completo
- **Optimización específica para keras.GRU** bidireccional
- **Soporte para señas expresivas** como GRACIAS, POR FAVOR
- **API MediaPipe Tasks moderna** (>=0.10.11)

#### 🏃‍♂️ **Métricas de Hombros (Nuevas):**
- Movimiento específico de hombros
- Simetría bilateral de hombros
- Coordinación mano-hombro temporal
- Amplitud de movimiento de torso superior

#### 🧠 **Optimizaciones para GRU:**
- Normalización en rango [-1, 1] ideal para GRU
- Información temporal rica con 9 velocidades
- Análisis de periodicidad y suavidad
- Criterios de calidad específicos para RNNs

#### 🎮 **Interfaz Mejorada:**
- HUD con información en tiempo real
- Indicadores visuales para manos y hombros
- Estado de tracking de pose
- Información específica para señas expresivas

#### 📊 **Sistema de Calidad Avanzado:**
- Evaluación específica por tipo de seña
- Criterios especiales para expresiones corporales
- Bonus por coordinación corporal
- Detección automática de problemas

### 🔧 **Mejoras Técnicas:**
- Procesamiento asíncrono de MediaPipe
- Gestión mejorada de memoria
- Manejo robusto de errores
- Compatibilidad con Python 3.11+

### 📈 **Comparativa de Versiones:**

| Característica | v1.0 | v2.0 |
|---|---|---|
| Features totales | 126 | **171** |
| Métricas calidad | 12 | **24** |
| Puntos pose | 8 | **12** |
| Velocidades | 2 | **9** |
| Frames/secuencia | 30 | **60** |
| Tracking hombros | ❌ | **✅** |
| Señas expresivas | ❌ | **✅** |
| API MediaPipe | Classic | **Tasks** |

### 🎯 **Señas Optimizadas en v2.0:**
- **GRACIAS**: Movimiento característico de hombros
- **POR FAVOR**: Coordinación mano-pecho-hombro
- **MUCHO GUSTO**: Expresión corporal completa
- **BUENOS DÍAS/NOCHES**: Apertura corporal expresiva

---

## [1.0.0] - 2025-06-XX

### 🎉 **Versión Inicial:**
- Recolector básico de datos LSP
- 126 features de manos únicamente
- 30 frames por secuencia
- 12 métricas de calidad básicas
- Soporte para alfabeto y palabras básicas
- API MediaPipe clásica
