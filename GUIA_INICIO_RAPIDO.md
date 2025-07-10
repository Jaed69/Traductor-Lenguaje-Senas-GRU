# 🎯 Guía de Inicio Rápido - LSP Traductor v2.0

## ⚡ **Instalación Rápida (5 minutos)**

### 1. **Clonar Repositorio:**
```bash
git clone https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU.git
cd Traductor-Lenguaje-Senas-GRU
```

### 2. **Instalar Dependencias:**
```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar paquetes
pip install -r requirements.txt
```

### 3. **Descargar Modelos MediaPipe:**
```bash
# Crear carpeta de modelos
mkdir models

# Descargar modelos (usar navegador o wget/curl)
# 1. hand_landmarker.task (21.8 MB)
# 2. pose_landmarker_heavy.task (12.7 MB)
```

**Enlaces de descarga:**
- [Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [Pose Landmarker](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task)

### 4. **Ejecutar:**
```bash
python data_c.py
```

## 🎮 **Primer Uso - Tutorial Interactivo**

### **Paso 1: Configurar cámara**
- Colócate a 1-1.5 metros de la cámara
- Iluminación uniforme (evitar contraluz)
- Fondo simple y contrastante

### **Paso 2: Recolectar primera seña**
1. Ejecutar `python data_c.py`
2. Seleccionar opción `1` (Recolectar seña específica)
3. Elegir seña `17. GRACIAS` (ideal para probar hombros)
4. Presionar `ESPACIO` para iniciar
5. Realizar seña durante 60 frames (~3 segundos)
6. Revisar calidad y aceptar/rechazar

### **Paso 3: Interpretar resultados**
```
📊 Calidad obtenida: EXCELENTE (Óptimo para GRU + Expresiones) (95.2%)
✅ Secuencia 0 guardada para 'GRACIAS'.
```

## 📊 **Entendiendo las Métricas**

### **HUD en Tiempo Real:**
- **GRABANDO (GRU + Hombros)**: Estado de grabación
- **Manos: 2 (Right, Left)**: Detección de manos
- **Features: 171**: Total de características extraídas
- **POSE: ON**: Tracking de hombros activo
- **Círculos verdes**: Tracking estable

### **Métricas de Calidad:**
- **92%+**: EXCELENTE - Óptimo para entrenar GRU
- **80-91%**: BUENA - Aceptable para GRU
- **65-79%**: REGULAR - Necesita mejora
- **<65%**: MALA - Descartar

## 🎭 **Señas Recomendadas para Empezar**

### **Principiantes:**
1. **A, B, C** - Alfabeto estático simple
2. **HOLA** - Movimiento dinámico básico
3. **AMOR** - Dos manos estáticas

### **Intermedio:**
4. **GRACIAS** - Expresión con hombros ⭐
5. **POR FAVOR** - Coordinación corporal ⭐
6. **J, Z** - Movimientos dinámicos complejos

### **Avanzado:**
7. **MUCHO GUSTO** - Expresión corporal completa ⭐
8. **BUENOS DÍAS** - Frase con componente corporal ⭐
9. **CÓMO ESTÁS** - Secuencia conversacional

*(⭐ = Optimizadas para tracking de hombros)*

## 🔧 **Solución de Problemas Comunes**

### **Error: "No se pudieron cargar los modelos"**
```bash
# Verificar que existan los archivos:
ls models/
# Debe mostrar:
# hand_landmarker.task
# pose_landmarker_heavy.task
```

### **Calidad baja constante:**
- ✅ Mejorar iluminación
- ✅ Fondo más simple
- ✅ Movimientos más lentos y deliberados
- ✅ Mantener manos en encuadre

### **Tracking de hombros no funciona:**
- ✅ Mostrar torso completo en cámara
- ✅ Usar ropa contrastante
- ✅ Evitar ropa muy holgada

### **"POSE: OFF" en HUD:**
- ✅ Alejarse más de la cámara
- ✅ Verificar que se vea torso completo
- ✅ Mejorar iluminación

## 📈 **Optimización de Datos para GRU**

### **Configuración Ideal:**
- **Secuencias**: 50 por seña mínimo
- **Calidad**: >80% para entrenamiento
- **Variabilidad**: Diferentes velocidades y amplitudes
- **Balance**: Igual cantidad por categoría

### **Dataset Recomendado:**
```
📁 Alfabeto Estático: 24 señas × 50 secuencias = 1,200
📁 Alfabeto Dinámico: 5 señas × 50 secuencias = 250
📁 Palabras Básicas: 4 señas × 50 secuencias = 200
📁 Expresiones: 5 señas × 50 secuencias = 250
📁 Frases: 4 señas × 50 secuencias = 200
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: 42 señas × 50 secuencias = 2,100 secuencias
```

## 🚀 **Siguientes Pasos**

1. **Recolectar dataset completo** (1-2 semanas)
2. **Entrenar modelo GRU** con keras
3. **Evaluar performance** con métricas
4. **Optimizar hiperparámetros**
5. **Desplegar aplicación**

## 💡 **Tips Profesionales**

### **Para Máxima Calidad:**
- 🕒 Recolectar en diferentes momentos del día
- 👥 Incluir múltiples personas (diversidad)
- 🎭 Variar expresividad y velocidad
- 📱 Usar diferentes cámaras si es posible

### **Para Eficiencia:**
- 🔄 Usar "Modo Mejora de Calidad" para optimizar
- 📊 Revisar estadísticas regularmente
- 🎯 Priorizar señas expresivas (GRACIAS, POR FAVOR)
- ⚡ Entrenar en lotes por categoría

---

## 🆘 **Soporte y Ayuda**

- 📖 **Documentación completa**: README.md
- 🐛 **Reportar bugs**: [GitHub Issues](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/issues)
- 💬 **Preguntas**: Crear Discussion en GitHub
- 📧 **Contacto directo**: twofigsthree@gmail.com

¡Listo para crear el mejor dataset de LSP optimizado para GRU! 🚀🎯
