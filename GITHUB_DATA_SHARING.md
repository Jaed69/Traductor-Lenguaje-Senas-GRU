# GitHub Data Sharing Setup

Este documento explica cómo configurar GitHub para compartir datos de entrenamiento del proyecto de traductor de lenguaje de señas.

## 🎯 Estrategia para GitHub

### Opción 1: Git LFS (Recomendada para archivos grandes)
Para archivos de datos pesados (>100MB), usamos Git LFS que permite almacenar archivos grandes en GitHub.

### Opción 2: Repositorio de Datos Separado
Crear un repositorio específico solo para datos, separado del código.

### Opción 3: Releases con Assets
Usar las releases de GitHub para distribuir datasets como archivos comprimidos.

## 🔧 Configuración de Git LFS

### Instalación de Git LFS
```bash
# Windows (usando chocolatey)
choco install git-lfs

# O descargar desde: https://git-lfs.github.io/
```

### Configuración en tu repositorio
```bash
# Inicializar Git LFS
git lfs install

# Rastrear archivos de datos
git lfs track "*.npy"
git lfs track "*.h5"
git lfs track "data/sequences/**/*"
git lfs track "shared_data/**/*.npy"

# Confirmar configuración
git add .gitattributes
git commit -m "Setup Git LFS for data files"
```

## 📁 Estructura para GitHub Data Sharing

```
MediaLengS/
├── data/                          # Datos locales (Git LFS)
│   ├── sequences/
│   ├── sign_model_gru.h5
│   └── label_encoder.npy
├── shared_data/                   # Datos compartidos (Git LFS)
│   ├── sample_dataset/           # Dataset de ejemplo
│   │   ├── user_001/
│   │   ├── user_002/
│   │   └── merged/
│   └── README.md
├── github_data_tools/            # Herramientas específicas para GitHub
│   ├── setup_github_sharing.py
│   ├── create_data_release.py
│   └── download_shared_data.py
└── docs/
    ├── GITHUB_DATA_SHARING.md    # Este archivo
    └── DATA_COLLECTION_GUIDE.md
```

## 🚀 Flujo de Trabajo con GitHub

### Para Contribuidores:

1. **Fork del repositorio principal**
   ```bash
   # Fork en GitHub UI, luego clonar
   git clone https://github.com/TU_USUARIO/MediaLengS.git
   cd MediaLengS
   git lfs install
   ```

2. **Agregar datos nuevos**
   ```bash
   # Recolectar datos
   python data_collector.py
   
   # Exportar para compartir
   python data_exporter.py --contributor-id user_XXX
   
   # Mover a shared_data
   mv shared_data/user_XXX shared_data/
   ```

3. **Commit y Push**
   ```bash
   git add shared_data/user_XXX/
   git commit -m "Add training data from user_XXX"
   git push origin main
   ```

4. **Pull Request**
   - Crear PR al repositorio principal
   - Incluir estadísticas de los datos en la descripción

### Para el Repositorio Principal:

1. **Revisar contribuciones**
   ```bash
   # Descargar datos de PR
   python github_data_tools/validate_contribution.py --pr-number 123
   ```

2. **Merge y Release**
   ```bash
   # Después de merge, crear release con dataset actualizado
   python github_data_tools/create_data_release.py --version v1.2.0
   ```

## 📦 Releases para Distribución

### Crear Release con Dataset
```bash
# Comprimir dataset actual
python github_data_tools/create_data_release.py --output dataset_v1.0.zip

# Subir como asset en GitHub Release
# - Ir a Releases en GitHub
# - Crear nueva release
# - Subir el archivo ZIP como asset
```

### Descargar Dataset de Release
```bash
# Para nuevos colaboradores
python github_data_tools/download_shared_data.py --version v1.0.0
```

## 🔍 Validación de Datos

### Pre-commit Hooks
```bash
# Instalar pre-commit
pip install pre-commit

# Configurar hooks para validar datos
pre-commit install
```

### Validación Automática
```yaml
# .github/workflows/validate-data.yml
name: Validate Data Contribution
on:
  pull_request:
    paths:
      - 'shared_data/**'
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          lfs: true
      - name: Validate Data Quality
        run: python dataset_stats.py --validate-only
```

## 📊 Dashboard de Datos

### GitHub Pages para Estadísticas
```bash
# Generar estadísticas como web
python dataset_stats.py --export-html docs/dataset_dashboard.html

# Configurar GitHub Pages para mostrar estadísticas públicas
```

## 🔐 Gestión de Acceso

### Repositorio Público vs Privado
- **Público**: Datos abiertos, máxima colaboración
- **Privado**: Control de acceso, colaboradores específicos

### Teams y Permisos
```bash
# Crear team para el proyecto
gh api orgs/TU_ORG/teams -f name="SignLanguage-Contributors"

# Agregar colaboradores
gh api orgs/TU_ORG/teams/sign-language-contributors/memberships/USERNAME
```

## 💰 Consideraciones de Costo

### Git LFS Limits (GitHub)
- **Gratis**: 1GB storage, 1GB bandwidth/mes
- **Pro**: 50GB storage, 50GB bandwidth/mes
- **Alternativa**: Git LFS con S3/Azure

### Optimización de Espacio
```bash
# Comprimir datos antes de commit
python github_data_tools/compress_datasets.py --input shared_data/
```

## 🛠️ Herramientas Específicas

### Scripts para GitHub Integration
1. `setup_github_sharing.py` - Configuración inicial
2. `validate_contribution.py` - Validar datos en PRs
3. `create_data_release.py` - Crear releases automáticamente
4. `download_shared_data.py` - Descargar datasets
5. `sync_with_github.py` - Sincronizar datos locales

## 📋 Mejores Prácticas

### Naming Convention
```
shared_data/
├── contributors/
│   ├── user_001_juan_perez/      # Formato: user_ID_nombre
│   ├── user_002_maria_garcia/
│   └── user_003_anonymous/
├── releases/
│   ├── v1.0.0_basic_signs/
│   ├── v1.1.0_extended_vocab/
│   └── v2.0.0_full_dataset/
└── samples/
    └── demo_dataset/             # Dataset pequeño para demos
```

### Commit Messages
```bash
git commit -m "data: Add 1200 sequences for user_001

- 40 sequences each for A-Z letters
- 200 sequences for HOLA, GRACIAS  
- Quality score: 8.5/10
- Camera: 1920x1080, good lighting"
```

### Documentation en PRs
- Estadísticas del dataset contribuido
- Condiciones de grabación
- Equipos utilizados
- Notas de calidad

## 🎉 Beneficios de GitHub Data Sharing

1. **🔄 Versionado**: Historial completo de cambios en datasets
2. **🤝 Colaboración**: Pull requests para revisar contribuciones
3. **📊 Transparencia**: Issues y discussions para coordinar
4. **🚀 CI/CD**: Validación automática de calidad
5. **📈 Métricas**: GitHub insights para ver actividad
6. **🌍 Accesibilidad**: Fácil descarga para nuevos colaboradores
7. **📝 Documentación**: Wiki y docs integradas
