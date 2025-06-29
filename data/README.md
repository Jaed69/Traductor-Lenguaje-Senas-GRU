# Directorio de Datos

Este directorio contiene los datos y modelos generados durante el entrenamiento del traductor de lenguaje de señas.

## Estructura

```
data/
├── sequences/          # Secuencias de datos recolectadas (se generan automáticamente)
│   ├── HOLA/          # Datos para la seña "HOLA"
│   ├── A/             # Datos para la letra "A"
│   └── ...            # Otras señas
├── sign_model_gru.h5  # Modelo entrenado (se genera automáticamente)
└── label_encoder.npy  # Codificador de etiquetas (se genera automáticamente)
```

## Nota Importante

Los archivos en este directorio se generan automáticamente cuando ejecutas:

1. `data_collector.py` - Genera las secuencias de datos
2. `model_trainer_sequence.py` - Genera el modelo entrenado y el codificador

**No necesitas crear estos archivos manualmente.** Simplemente ejecuta los scripts en orden y se generarán automáticamente.
