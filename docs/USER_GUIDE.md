# 🎯 Guía de Usuario LSP v2.0

## 📋 Índice
- [Primeros Pasos](#primeros-pasos)
- [Recolección de Datos](#recolección-de-datos)
- [Data Augmentation](#data-augmentation)
- [Entrenamiento](#entrenamiento)
- [Inferencia](#inferencia)
- [Consejos y Mejores Prácticas](#consejos-y-mejores-prácticas)

## Primeros Pasos

### 🚀 Inicio del Sistema

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Ejecutar sistema principal
python run.py
```

### 📱 Menú Principal

Al iniciar, verás el menú principal:

```
🚀 SISTEMA LSP v2.0 - TRADUCTOR DE LENGUAJE DE SEÑAS
═══════════════════════════════════════════════════

Selecciona una opción:
[1] 📊 Recolección de Datos
[2] 🧠 Entrenamiento de Modelo  
[3] 📈 Evaluación de Modelo
[4] 🎯 Inferencia en Tiempo Real
[5] ⚙️ Configuración del Sistema
[Q] 🚪 Salir

Ingresa tu opción: 
```

## Recolección de Datos

### 📊 Menú de Recolección

Al seleccionar opción [1], accederás al módulo de recolección:

```
🚀 RECOLECTOR DE DATOS LSP - VERSIÓN MODULAR v2.0
═══════════════════════════════════════════════════

📊 PROGRESO GENERAL DEL DATASET:
   📈 Progreso total: 47/190 secuencias (24.7%)
   ✅ Señas completadas: 0/5
   ⚠️ Secuencias faltantes: 143
   📊 [█████████░░░░░░░░░░░░░░░░░░░] 24.7%

📋 Señas disponibles:
   ⚠️  1. A     [3/30] (faltan 27) - 🔤 LETRA ESTÁTICA
   ⚠️  2. HOLA  [24/50] (faltan 26) - 💬 PALABRA
   ✅  3. B     [30/30] (completa) - 🔤 LETRA ESTÁTICA
   ...

Opciones especiales:
   [ALL] - 🎯 Recolectar todas las señas pendientes
   [A]   - 🔄 Data Augmentation automático
   [S]   - 📊 Ver estadísticas detalladas
   [Q]   - 🚪 Volver al menú principal

Ingresa tu opción (número, ALL, A, S, Q): 
```

### 🎯 Recolección Individual

#### Proceso de Recolección:

1. **Selección de Seña**: Ingresa el número de la seña (ej: `1` para la seña A)

2. **Preparación**: El sistema muestra las instrucciones:
   ```
   📋 RECOLECTANDO SEÑA: A
   ═══════════════════════
   
   📝 Instrucciones:
   • Forma la letra A con tu mano dominante
   • Mantén la posición estable durante la grabación
   • Asegúrate de tener buena iluminación
   
   💡 Consejos:
   • Usa tu mano dominante
   • Mantén la seña centrada en la cámara
   • Evita movimientos bruscos
   
   📊 Progreso: 3/30 secuencias recolectadas
   
   Presiona ENTER para comenzar (o 'q' para cancelar):
   ```

3. **Grabación**: 
   - La cámara se activa automáticamente
   - Cuenta regresiva de 3 segundos
   - Grabación de 2 segundos (60 frames)
   - Análisis automático de calidad

4. **Validación de Calidad**:
   ```
   ✅ SECUENCIA RECOLECTADA EXITOSAMENTE
   ═══════════════════════════════════
   
   📊 Análisis de Calidad:
   • Confianza de landmarks: 94.2% ✅
   • Estabilidad de manos: 91.7% ✅
   • Claridad de seña: 88.5% ✅
   • Calidad general: EXCELENTE ⭐⭐⭐
   
   📈 Progreso actualizado: 4/30 (13.3%)
   
   ¿Continuar recolectando? (y/n):
   ```

### 🎯 Recolección Masiva (ALL)

Para recolectar todas las señas pendientes:

1. Selecciona `ALL` en el menú
2. El sistema calcula automáticamente las señas pendientes
3. Recolección automática con pausas entre señas

```
🎯 RECOLECCIÓN AUTOMÁTICA DE TODAS LAS SEÑAS
═══════════════════════════════════════════

📊 Resumen de trabajo:
• Total de señas pendientes: 4
• Secuencias por recolectar: 143
• Tiempo estimado: 47 minutos

🔄 Orden de recolección:
1. A (27 secuencias pendientes)
2. HOLA (26 secuencias pendientes)
3. C (30 secuencias pendientes)
4. GRACIAS (30 secuencias pendientes)

¿Proceder con recolección automática? (y/n):
```

## Data Augmentation

### 🔄 Acceso a Data Augmentation

Selecciona `A` en el menú de recolección para acceder:

```
🔄 DATA AUGMENTATION LSP v2.0
═══════════════════════════════

📊 Estado actual del dataset:
• Total de secuencias originales: 47
• Potencial de augmentación: 188 secuencias adicionales
• Reducción de trabajo manual: ~70%

🎯 Opciones de augmentación:
[1] 🚀 Augmentación automática (todas las señas)
[2] 🎯 Augmentación selectiva (señas específicas)
[3] 📊 Análisis de potencial de augmentación
[4] ⚙️ Configuración avanzada
[Q] 🚪 Volver al menú anterior

Selecciona una opción:
```

### 🚀 Augmentación Automática

Proceso completamente automatizado:

1. **Análisis Inicial**:
   ```
   🔍 ANÁLISIS DE DATASET PARA AUGMENTACIÓN
   ═══════════════════════════════════════
   
   📊 Señas disponibles para augmentación:
   ✅ A (3 secuencias) → puede generar 9 adicionales
   ✅ HOLA (24 secuencias) → puede generar 72 adicionales  
   ✅ B (30 secuencias) → puede generar 90 adicionales
   
   🎯 Total augmentable: 47 → 188 secuencias (+300%)
   ```

2. **Configuración Automática**:
   ```
   ⚙️ CONFIGURACIÓN AUTOMÁTICA
   ═══════════════════════════
   
   🔄 Técnicas seleccionadas por tipo:
   • Letras estáticas: Espacial + Ruido + Manos
   • Palabras: Temporal + Espacial + Ruido
   • Frases: Temporal ligero + Ruido
   
   📊 Factor de augmentación: 3x
   🎯 Umbral de calidad: 85%
   ```

3. **Procesamiento**:
   ```
   🚀 PROCESANDO AUGMENTACIÓN...
   
   [A] Procesando seña A...
   ████████████████████ 100% (3/3 secuencias)
   ✅ Generadas: 9 secuencias (calidad promedio: 87.2%)
   
   [HOLA] Procesando seña HOLA...
   ████████████████████ 100% (24/24 secuencias)
   ✅ Generadas: 72 secuencias (calidad promedio: 89.1%)
   
   📊 RESUMEN FINAL:
   • Secuencias originales: 47
   • Secuencias generadas: 171
   • Total final: 218 secuencias
   • Tiempo ahorrado: ~34 horas de recolección manual
   ```

### 🎯 Augmentación Selectiva

Para control granular por seña:

1. **Selección de Señas**:
   ```
   🎯 AUGMENTACIÓN SELECTIVA
   ════════════════════════
   
   📋 Selecciona señas para augmentar:
   [1] A (3 secuencias)
   [2] HOLA (24 secuencias)
   [3] B (30 secuencias)
   [ALL] Todas las señas
   
   Ingresa números separados por comas (ej: 1,3):
   ```

2. **Configuración Personalizada**:
   ```
   ⚙️ CONFIGURACIÓN PARA SEÑAS SELECCIONADAS
   ═══════════════════════════════════════
   
   🔄 Factor de augmentación:
   [1] Conservadora (2x) - Alta calidad
   [2] Moderada (3x) - Balance calidad/cantidad  
   [3] Agresiva (5x) - Máxima expansión
   [4] Personalizada
   
   Selecciona opción [1-4]:
   ```

## Entrenamiento

### 🧠 Módulo de Entrenamiento

Selecciona opción [2] del menú principal:

```
🧠 ENTRENAMIENTO DE MODELO GRU
═════════════════════════════

📊 Estado del Dataset:
• Total de secuencias: 218
• Señas únicas: 5
• Formato: Keras/TensorFlow
• Calidad promedio: 88.7%

⚙️ Configuración de Entrenamiento:
• Arquitectura: GRU Bidireccional
• Secuencias: 60 frames
• Features por frame: 157
• Epochs: 100
• Batch size: 32

🎯 Opciones:
[1] 🚀 Entrenamiento automático (configuración optimizada)
[2] ⚙️ Entrenamiento personalizado
[3] 📊 Análisis de dataset
[4] 🔄 Continuar entrenamiento previo
[Q] 🚪 Volver al menú principal

Selecciona una opción:
```

### 🚀 Entrenamiento Automático

Proceso optimizado para mejores resultados:

```
🚀 INICIANDO ENTRENAMIENTO AUTOMÁTICO
═══════════════════════════════════

📊 Preparación de datos:
✅ Cargando dataset... (218 secuencias)
✅ División train/validation/test: 70%/15%/15%
✅ Normalización de features aplicada
✅ Augmentación temporal activada

🧠 Arquitectura del modelo:
┌─────────────────────────────────┐
│ Input: (None, 60, 157)          │
│ ↓                               │
│ GRU Bidireccional (128 units)   │
│ ↓                               │
│ Dropout (0.3)                   │
│ ↓                               │
│ GRU Bidireccional (64 units)    │
│ ↓                               │
│ Dense (32, ReLU)                │
│ ↓                               │
│ Output: (5 clases)              │
└─────────────────────────────────┘

🚀 Entrenando modelo...
Epoch 1/100: loss: 1.234 - accuracy: 0.234 - val_accuracy: 0.189
Epoch 2/100: loss: 0.987 - accuracy: 0.456 - val_accuracy: 0.423
...
```

## Inferencia

### 🎯 Traducción en Tiempo Real

Selecciona opción [4] del menú principal:

```
🎯 INFERENCIA EN TIEMPO REAL
═══════════════════════════

📊 Estado del Sistema:
✅ Modelo cargado: lsp_model_v2.h5 (Precisión: 94.2%)
✅ MediaPipe inicializado
✅ Cámara detectada: 1080p
✅ Sistema listo para traducir

🎯 Opciones de Inferencia:
[1] 🎥 Traducción continua (recomendado)
[2] 📸 Traducción por capturas
[3] 📁 Traducir desde archivo de video
[4] ⚙️ Configurar parámetros de inferencia
[Q] 🚪 Volver al menú principal

Selecciona una opción:
```

### 🎥 Traducción Continua

Modo principal de traducción:

```
🎥 TRADUCCIÓN CONTINUA ACTIVADA
═════════════════════════════

📋 Instrucciones:
• Realiza señas frente a la cámara
• El sistema traducirá automáticamente
• Presiona 'q' para salir

🎯 Configuración actual:
• Confianza mínima: 70%
• Buffer de frames: 60
• Suavizado temporal: Activado

Presiona ENTER para comenzar...

═══════════════════════════════════════
         🎥 TRADUCCIÓN EN VIVO
═══════════════════════════════════════

Señas detectadas:
[14:23:45] HOLA (Confianza: 87.3%) ⭐⭐⭐
[14:23:48] A (Confianza: 91.2%) ⭐⭐⭐
[14:23:52] GRACIAS (Confianza: 84.7%) ⭐⭐⭐

📊 Estadísticas de sesión:
• Tiempo de sesión: 2m 34s
• Señas detectadas: 12
• Precisión promedio: 88.1%
• FPS promedio: 28.4

═══════════════════════════════════════
```

## Consejos y Mejores Prácticas

### 🎯 Para Recolección de Datos

#### ✅ Configuración Óptima
- **Iluminación**: Uniforme, sin sombras fuertes
- **Fondo**: Neutro, preferiblemente liso
- **Distancia**: 1-2 metros de la cámara
- **Posición**: Centrado en el encuadre

#### ✅ Técnica de Señas
- **Estabilidad**: Mantén posición estable en letras estáticas
- **Fluidez**: Movimientos naturales en palabras/frases
- **Consistencia**: Repite señas de manera similar
- **Visibilidad**: Asegura que ambas manos sean visibles

#### ✅ Calidad del Dataset
```
Excelente (90%+):  ⭐⭐⭐ - Usar para entrenamiento principal
Buena (80-89%):    ⭐⭐   - Usar con augmentación
Aceptable (70-79%): ⭐    - Considerar recolectar nuevamente
Baja (<70%):       ❌     - Recolectar nuevamente
```

### 🔄 Para Data Augmentation

#### ✅ Cuándo Usar Augmentación
- **Dataset pequeño**: < 30 secuencias por seña
- **Desequilibrio**: Algunas señas con pocas muestras
- **Tiempo limitado**: Necesidad de expandir rápidamente
- **Validación**: Crear conjunto de validación robusto

#### ✅ Configuración por Escenario
```
Investigación/Prototipo:
• Factor: 5x
• Calidad: 80%
• Técnicas: Todas

Producción:
• Factor: 3x  
• Calidad: 90%
• Técnicas: Conservadoras

Demo/Presentación:
• Factor: 2x
• Calidad: 95%
• Técnicas: Mínimas
```

### 🧠 Para Entrenamiento

#### ✅ Preparación de Datos
- **Mínimo recomendado**: 50 secuencias por seña
- **Óptimo**: 100+ secuencias por seña
- **Distribución**: Balanceada entre todas las señas
- **Calidad**: 85%+ promedio del dataset

#### ✅ Configuración del Modelo
```
Dataset pequeño (< 1000 secuencias):
• GRU units: 64, 32
• Dropout: 0.5
• Learning rate: 0.001
• Epochs: 50

Dataset mediano (1000-5000 secuencias):
• GRU units: 128, 64  
• Dropout: 0.3
• Learning rate: 0.0005
• Epochs: 100

Dataset grande (> 5000 secuencias):
• GRU units: 256, 128
• Dropout: 0.2
• Learning rate: 0.0001
• Epochs: 150
```

### 🎯 Para Inferencia

#### ✅ Optimización de Performance
- **Resolución**: 720p para balance speed/calidad
- **FPS**: 30 FPS para suavidad óptima
- **Buffer**: 60 frames para contexto temporal
- **Confianza**: 70%+ para predicciones confiables

#### ✅ Mejores Prácticas de Uso
- **Calibración**: Realizar señas de prueba antes de uso serio
- **Consistencia**: Mantener posición y estilo similares al entrenamiento
- **Paciencia**: Permitir al sistema procesar secuencias completas
- **Retroalimentación**: Usar métricas de confianza para validar

### 📊 Monitoreo y Mantenimiento

#### ✅ Métricas Clave
```
Recolección:
• Secuencias por hora: 15-20
• Calidad promedio: >85%
• Tasa de rechazo: <10%

Entrenamiento:
• Accuracy de validación: >90%
• Loss convergencia: <0.1
• Overfitting: Diferencia <5% train/val

Inferencia:
• FPS: >25
• Latencia: <100ms
• Confianza promedio: >80%
```

#### ✅ Mantenimiento Regular
- **Backup**: Respaldar datos y modelos semanalmente
- **Limpieza**: Revisar y limpiar datos de baja calidad
- **Actualización**: Reentrenar modelo con nuevos datos
- **Validación**: Probar sistema con usuarios reales

---

## 🆘 Soporte y Ayuda

### 📞 Recursos de Ayuda
- **Documentación completa**: [README.md](../README.md)
- **Guías técnicas**: [docs/](../docs/)
- **Issues y soporte**: [GitHub Issues](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/issues)

### 🔧 Solución de Problemas
- **Guía de instalación**: [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- **Diagnóstico del sistema**: `python diagnosis.py`
- **Logs del sistema**: `logs/lsp_system.log`

---

**🎯 ¡Disfruta usando el Sistema LSP v2.0!** 

*Democratizando el acceso al Lenguaje de Señas Peruano a través de la tecnología.*
