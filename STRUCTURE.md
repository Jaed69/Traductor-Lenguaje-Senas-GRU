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
│   └── label_encoder.npy         # Codificador de etiquetas (generado automáticamente)
│
├── data_collector.py             # Script para recolectar datos de entrenamiento
├── model_trainer_sequence.py     # Script para entrenar el modelo GRU
├── real_time_translator.py       # Script principal para traducción en tiempo real
├── main.py                       # Punto de entrada principal del proyecto
│
├── requirements.txt              # Dependencias específicas (completo)
├── requirements_clean.txt        # Dependencias principales (limpio)
├── README.md                     # Documentación del proyecto
├── LICENSE                       # Licencia MIT
├── .gitignore                    # Archivos a ignorar en Git
└── STRUCTURE.md                  # Este archivo
```

## Flujo de Trabajo

1. **Recolección de Datos**: Ejecutar `data_collector.py`
2. **Entrenamiento**: Ejecutar `model_trainer_sequence.py`
3. **Traducción**: Ejecutar `main.py` o `real_time_translator.py`

## Archivos Principales

- **data_collector.py**: Interfaz para capturar secuencias de señas usando la cámara web
- **model_trainer_sequence.py**: Entrena un modelo GRU con las secuencias capturadas
- **real_time_translator.py**: Realiza predicciones en tiempo real usando el modelo entrenado
- **main.py**: Script principal que inicializa el traductor
