# Estructura del Proyecto

```
MediaLengS/
│
├── data/                          # Directorio para datos y modelos (no se sube a Git)
│   ├── sequences/                 # Secuencias de datos recolectadas
│   │   ├── HOLA/                 # Ejemplo: datos para la seña "HOLA"
│   │   ├── A/                    # Ejemplo: datos para la letra "A"
│   │   └── ...                   # Otras señas
│   ├── sign_model_gru.h5         # Modelo entrenado (generado automáticamente)
│   ├── label_encoder.npy         # Codificador de etiquetas (generado automáticamente)
│   └── README.md                 # Documentación del directorio de datos
│
├── shared_data/                   # Directorio para datos compartidos (colaboración)
│   ├── user_001/                 # Datos del contribuidor 1
│   ├── user_002/                 # Datos del contribuidor 2
│   ├── merged/                   # Dataset combinado
│   └── backups/                  # Respaldos de datos
│
├── collaboration_config/          # Configuración para colaboración
│   ├── collaboration_config.json # Configuración principal
│   ├── contributor_template.json # Plantilla para contribuidores
│   └── quality_guidelines.md     # Guías de calidad
│
├── dataset_analysis/             # Análisis y visualizaciones del dataset
│   ├── distribucion_senas.png    # Gráfico de distribución
│   ├── calidad_senas.png         # Gráfico de calidad
│   └── dataset_report.txt        # Reporte de análisis
│
├── data_collector.py             # Script para recolectar datos de entrenamiento
├── model_trainer_sequence.py     # Script para entrenar el modelo GRU
├── real_time_translator.py       # Script principal para traducción en tiempo real
├── main.py                       # Punto de entrada principal del proyecto
│
├── data_exporter.py              # Exportar datos para compartir
├── data_importer.py              # Importar datos de colaboradores
├── dataset_stats.py              # Análisis estadístico del dataset
├── setup_collaboration.py        # Configuración de herramientas colaborativas
│
├── requirements.txt              # Dependencias específicas (completo)
├── requirements_clean.txt        # Dependencias principales (limpio)
├── README.md                     # Documentación del proyecto
├── DATA_SHARING.md               # Guía de colaboración y compartición de datos
├── DEVELOPMENT.md                # Guía de desarrollo
├── STRUCTURE.md                  # Este archivo
├── LICENSE                       # Licencia MIT
└── .gitignore                    # Archivos a ignorar en Git
```

## Flujo de Trabajo

### Desarrollo Individual
1. **Recolección de Datos**: Ejecutar `data_collector.py`
2. **Entrenamiento**: Ejecutar `model_trainer_sequence.py`
3. **Traducción**: Ejecutar `main.py` o `real_time_translator.py`

### Desarrollo Colaborativo
1. **Configuración Inicial**: Ejecutar `setup_collaboration.py`
2. **Exportar Datos**: `data_exporter.py --contributor-id user_XXX`
3. **Compartir**: Subir datos a repositorio/drive compartido
4. **Importar**: `data_importer.py --import shared_data/user_XXX/`
5. **Análisis**: `dataset_stats.py --visualizations`
6. **Entrenamiento Colaborativo**: `model_trainer_sequence.py --use-merged-data`

## Archivos Principales

### Scripts de Datos
- **data_collector.py**: Interfaz para capturar secuencias de señas usando la cámara web
- **data_exporter.py**: Exporta datos locales en formato estándar para compartir
- **data_importer.py**: Importa y combina datos de otros colaboradores
- **dataset_stats.py**: Genera estadísticas y visualizaciones del dataset

### Scripts de Modelo
- **model_trainer_sequence.py**: Entrena un modelo GRU con las secuencias capturadas
- **real_time_translator.py**: Realiza predicciones en tiempo real usando el modelo entrenado
- **main.py**: Script principal que inicializa el traductor

### Scripts de Colaboración
- **setup_collaboration.py**: Configura automáticamente las herramientas colaborativas

## Características de Colaboración

### 🔄 Compartición de Datos
- Formato estándar de intercambio
- Metadatos de calidad y contribuidor
- Validación automática de datos importados

### 📊 Análisis de Calidad
- Métricas de calidad por contribuidor
- Detección de datos de baja calidad
- Visualizaciones de distribución del dataset

### 🎯 Escalabilidad
- Support para datasets de múltiples usuarios
- Entrenamiento con datos combinados
- Herramientas de análisis estadístico

### 🛡️ Privacidad
- Solo se comparten landmarks (no imágenes)
- Anonimización opcional de contribuidores
- Control sobre datos personales
