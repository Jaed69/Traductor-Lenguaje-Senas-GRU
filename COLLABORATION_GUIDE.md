# Guía de Colaboración GitHub

## 🎯 Para Colaboradores Nuevos

### 1. Fork y Clona el Repositorio
```bash
# 1. Fork el repo en GitHub (botón Fork)
# 2. Clona tu fork
git clone https://github.com/TU_USUARIO/MediaLengS.git
cd MediaLengS

# 3. Configura el upstream
git remote add upstream https://github.com/USUARIO_ORIGINAL/MediaLengS.git

# 4. Inicializa Git LFS
git lfs install
```

### 2. Recolecta y Contribuye Datos
```bash
# Activa el entorno
conda activate tf_gpu

# Recolecta tus datos
python data_collector.py

# Exporta para compartir
python data_exporter.py --contributor-id user_XXX --contributor-name "Tu Nombre"

# Los datos se exportan a shared_data/contributors/user_XXX/
```

### 3. Crea Pull Request
```bash
# Agrega los datos
git add shared_data/contributors/user_XXX/
git commit -m "data: Add training sequences from user_XXX

- XX sequences for HOLA, A, B, C signs
- Quality score: X.X/10
- Camera: resolución, lighting conditions"

# Push a tu fork
git push origin main

# Crea PR en GitHub UI
```

## 🔄 Para Usar Datos de Otros

### Descargar Dataset Completo
```bash
# Descarga la última release
python github_data_tools/download_shared_data.py --version latest

# O específica
python github_data_tools/download_shared_data.py --version v1.0.0
```

### Importar Datos Específicos
```bash
# Después de un git pull con nuevas contribuciones
python data_importer.py --import shared_data/contributors/user_XXX/
```

### Entrenar con Datos Combinados
```bash
# Entrena con todos los datos disponibles
python model_trainer_sequence.py --use-merged-data --epochs 100
```

## 📊 Analizar Dataset
```bash
# Análisis completo con gráficos
python dataset_stats.py --visualizations --report github_dataset_report.txt

# Ver estadísticas rápidas
python dataset_stats.py
```

## 🚀 Crear Release de Dataset
```bash
# Para maintainers del repo principal
python github_data_tools/create_data_release.py --version v1.1.0 --description "Extended dataset with A-Z letters"

# Subir a GitHub Release manualmente o con GitHub CLI
gh release create v1.1.0 releases/dataset_v1.1.0.zip --title "Dataset v1.1.0"
```

## 📋 Estructura de Contribución

```
shared_data/contributors/user_XXX/
├── metadata.json          # Información del contribuidor
├── README.md             # Documentación
└── sequences/            # Datos de secuencias
    ├── HOLA/
    ├── A/
    └── ...
```

## 🎯 Beneficios GitHub vs Métodos Tradicionales

| Aspecto | Método Tradicional | GitHub Data Sharing |
|---------|-------------------|-------------------|
| **Versionado** | Manual | Automático con Git |
| **Colaboración** | Email/Drive | Pull Requests |
| **Validación** | Manual | GitHub Actions |
| **Distribución** | Enlaces rotos | Releases estables |
| **Documentación** | Dispersa | Integrada en repo |
| **Acceso** | Permisos complejos | Fork simple |

## 💡 Mejores Prácticas

### Nombres de Commit
```bash
git commit -m "data: Add 200 sequences from user_003

- 40 sequences each for A, B, C, D, E
- Quality score: 8.2/10
- Camera: 1920x1080, professional lighting
- Background: uniform white"
```

### Pull Request Template
```markdown
## Contribución de Datos

### Información del Contribuidor
- **ID**: user_XXX
- **Nombre**: [Tu Nombre]
- **Total secuencias**: XXX

### Estadísticas
- **Señas contribuidas**: A, B, C, HOLA
- **Secuencias por seña**: 40 promedio
- **Calidad promedio**: X.X/10

### Condiciones de Grabación
- **Cámara**: resolución
- **Iluminación**: buena/regular/mala
- **Fondo**: uniforme/variable/complejo

### Notas Adicionales
[Cualquier información relevante]
```

## 🔧 Troubleshooting

### Error de Git LFS
```bash
# Si Git LFS no funciona
git lfs install --force
git lfs pull
```

### Archivos Muy Pesados
```bash
# Comprimir antes de subir
python github_data_tools/compress_datasets.py --input shared_data/contributors/user_XXX/
```

### Conflictos de Merge
```bash
# Actualizar desde upstream
git fetch upstream
git merge upstream/main
# Resolver conflictos manualmente
```

## 📈 Roadmap de Colaboración

### Fase Actual: v1.0 ✅
- [x] Git LFS configurado
- [x] Estructura de shared_data
- [x] Scripts de export/import
- [x] Dataset inicial con HOLA

### Fase 2: v1.1 🔄
- [ ] 5+ contribuidores
- [ ] Dataset completo A-Z
- [ ] Validación automática de calidad
- [ ] Dashboard web de estadísticas

### Fase 3: v2.0 🚀
- [ ] 10+ contribuidores
- [ ] Palabras completas (HOLA, GRACIAS, etc.)
- [ ] Modelo mejorado con datos diversos
- [ ] API para descarga automática

¡Únete a la colaboración y ayuda a crear el mejor dataset de lenguaje de señas! 🤟
