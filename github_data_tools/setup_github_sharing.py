# setup_github_sharing.py
"""
Script para configurar GitHub como plataforma de compartición de datos.
Configura Git LFS y prepara el repositorio para colaboración con datos pesados.
"""

import os
import subprocess
import json
from datetime import datetime

class GitHubDataSharingSetup:
    def __init__(self):
        self.repo_root = os.getcwd()
        self.gitattributes_path = os.path.join(self.repo_root, '.gitattributes')
        
    def check_git_lfs_installed(self):
        """Verifica si Git LFS está instalado"""
        try:
            result = subprocess.run(['git', 'lfs', 'version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Git LFS está instalado")
                print(f"   Versión: {result.stdout.strip()}")
                return True
            else:
                print("❌ Git LFS no está instalado")
                return False
        except FileNotFoundError:
            print("❌ Git LFS no está instalado")
            return False
    
    def install_git_lfs_instructions(self):
        """Muestra instrucciones para instalar Git LFS"""
        print("\n🔧 INSTRUCCIONES PARA INSTALAR GIT LFS:")
        print("="*50)
        print("\n📥 Opción 1 - Descarga directa:")
        print("   1. Ve a: https://git-lfs.github.io/")
        print("   2. Descarga el instalador para Windows")
        print("   3. Ejecuta el instalador")
        
        print("\n📦 Opción 2 - Usando Chocolatey:")
        print("   choco install git-lfs")
        
        print("\n📦 Opción 3 - Usando Scoop:")
        print("   scoop install git-lfs")
        
        print("\n🔄 Después de instalar, ejecuta este script nuevamente")
    
    def setup_git_lfs(self):
        """Configura Git LFS en el repositorio"""
        print("\n🔧 Configurando Git LFS...")
        
        try:
            # Inicializar Git LFS
            result = subprocess.run(['git', 'lfs', 'install'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Git LFS inicializado en el repositorio")
            else:
                print(f"❌ Error inicializando Git LFS: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error configurando Git LFS: {e}")
            return False
        
        return True
    
    def create_gitattributes(self):
        """Crea o actualiza .gitattributes para archivos de datos"""
        print("\n📝 Configurando .gitattributes para archivos de datos...")
        
        lfs_rules = [
            "# Git LFS tracking for data files",
            "*.npy filter=lfs diff=lfs merge=lfs -text",
            "*.h5 filter=lfs diff=lfs merge=lfs -text",
            "*.pkl filter=lfs diff=lfs merge=lfs -text",
            "*.zip filter=lfs diff=lfs merge=lfs -text",
            "",
            "# Track entire data directories",
            "data/sequences/** filter=lfs diff=lfs merge=lfs -text",
            "shared_data/**/*.npy filter=lfs diff=lfs merge=lfs -text",
            "shared_data/**/*.h5 filter=lfs diff=lfs merge=lfs -text",
            "",
            "# Dataset releases",
            "releases/*.zip filter=lfs diff=lfs merge=lfs -text",
            "releases/*.tar.gz filter=lfs diff=lfs merge=lfs -text"
        ]
        
        # Leer .gitattributes existente si existe
        existing_content = []
        if os.path.exists(self.gitattributes_path):
            with open(self.gitattributes_path, 'r') as f:
                existing_content = f.read().splitlines()
        
        # Verificar si ya están configuradas las reglas LFS
        has_lfs_rules = any("filter=lfs" in line for line in existing_content)
        
        if not has_lfs_rules:
            # Agregar reglas LFS
            with open(self.gitattributes_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join([''] + lfs_rules))
            print("✅ Reglas Git LFS agregadas a .gitattributes")
        else:
            print("✅ .gitattributes ya contiene reglas Git LFS")
    
    def create_shared_data_structure(self):
        """Crea la estructura de directorios para datos compartidos"""
        print("\n📁 Creando estructura de directorios...")
        
        directories = [
            "shared_data",
            "shared_data/contributors",
            "shared_data/releases", 
            "shared_data/samples",
            "shared_data/samples/demo_dataset",
            "releases"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
            # Crear README en cada directorio
            readme_path = os.path.join(directory, "README.md")
            if not os.path.exists(readme_path):
                self._create_directory_readme(directory, readme_path)
        
        print("✅ Estructura de directorios creada")
    
    def _create_directory_readme(self, directory, readme_path):
        """Crea README específico para cada directorio"""
        content = ""
        
        if "contributors" in directory:
            content = """# Contributors Data

Este directorio contiene datos de entrenamiento de diferentes contribuidores.

## Estructura
- `user_XXX_nombre/` - Datos de cada contribuidor
- Cada carpeta incluye metadatos y secuencias de entrenamiento

## Para agregar tus datos:
1. `python data_exporter.py --contributor-id user_XXX`
2. Mover la carpeta generada aquí
3. Hacer commit y push con Git LFS
"""
        elif "releases" in directory:
            content = """# Dataset Releases

Datasets compilados para distribución.

## Formato de Nombres:
- `vX.Y.Z_descripcion/` - Versión y descripción
- Incluye dataset combinado y metadatos

## Para crear release:
`python github_data_tools/create_data_release.py --version vX.Y.Z`
"""
        elif "samples" in directory:
            content = """# Sample Datasets

Datasets pequeños para demostración y pruebas.

## Contenido:
- `demo_dataset/` - Dataset básico para testing
- Datos ligeros que no requieren Git LFS
"""
        else:
            content = f"""# {directory.replace('_', ' ').title()}

Directorio para {directory.replace('_', ' ')} del proyecto.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def create_github_workflows(self):
        """Crea workflows de GitHub Actions para validación de datos"""
        print("\n⚙️ Creando GitHub Actions workflows...")
        
        workflows_dir = os.path.join('.github', 'workflows')
        os.makedirs(workflows_dir, exist_ok=True)
        
        # Workflow para validar datos en PR
        validate_workflow = """name: Validate Data Contribution

on:
  pull_request:
    paths:
      - 'shared_data/**'
      - 'data/**'

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          lfs: true
          
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          pip install numpy scikit-learn matplotlib
          
      - name: Validate data quality
        run: |
          python dataset_stats.py --validate-only
          
      - name: Generate data report
        run: |
          python dataset_stats.py --report pr_data_report.txt
          
      - name: Comment PR with results
        uses: actions/github-script@v6
        if: always()
        with:
          script: |
            const fs = require('fs');
            if (fs.existsSync('pr_data_report.txt')) {
              const report = fs.readFileSync('pr_data_report.txt', 'utf8');
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: '## 📊 Data Quality Report\\n\\n```\\n' + report + '\\n```'
              });
            }
"""
        
        workflow_path = os.path.join(workflows_dir, 'validate-data.yml')
        with open(workflow_path, 'w', encoding='utf-8') as f:
            f.write(validate_workflow)
        
        print("✅ GitHub Actions workflow creado")
    
    def create_data_sharing_config(self):
        """Crea configuración para compartición de datos"""
        config = {
            "github_data_sharing": {
                "enabled": True,
                "setup_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "git_lfs_enabled": True,
                "max_file_size_mb": 100,
                "supported_formats": [".npy", ".h5", ".pkl", ".zip"],
                "quality_threshold": 5.0,
                "auto_validation": True
            },
            "repository_settings": {
                "lfs_bandwidth_limit_gb": 1,
                "storage_limit_gb": 1,
                "contributors_limit": 10
            },
            "workflows": {
                "validate_on_pr": True,
                "auto_release": False,
                "quality_checks": True
            }
        }
        
        config_path = os.path.join('github_data_tools', 'github_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuración de GitHub data sharing creada")
    
    def show_next_steps(self):
        """Muestra los próximos pasos para el usuario"""
        print("\n" + "="*60)
        print("🎉 ¡CONFIGURACIÓN DE GITHUB DATA SHARING COMPLETADA!")
        print("="*60)
        
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Hacer commit de la configuración:")
        print("   git add .gitattributes .github/ github_data_tools/")
        print("   git commit -m 'Setup GitHub data sharing with Git LFS'")
        
        print("\n2. Push al repositorio:")
        print("   git push origin main")
        
        print("\n3. Subir datos existentes:")
        print("   git add data/ shared_data/")
        print("   git commit -m 'Add training data with Git LFS'")
        print("   git push origin main")
        
        print("\n4. Para que colaboradores contribuyan:")
        print("   - Fork del repositorio")
        print("   - Agregar datos en shared_data/contributors/")
        print("   - Pull request con los nuevos datos")
        
        print("\n📖 DOCUMENTACIÓN:")
        print("   - GITHUB_DATA_SHARING.md: Guía completa")
        print("   - github_data_tools/: Scripts de utilidad")
        
        print("\n⚠️  IMPORTANTE:")
        print("   - Git LFS tiene límites de ancho de banda")
        print("   - Plan gratuito: 1GB storage, 1GB bandwidth/mes")
        print("   - Considera upgrade si necesitas más espacio")

def main():
    print("🚀 CONFIGURANDO GITHUB PARA COMPARTICIÓN DE DATOS")
    print("Traductor de Lenguaje de Señas - GitHub Data Sharing Setup")
    print("="*70)
    
    setup = GitHubDataSharingSetup()
    
    # Verificar Git LFS
    if not setup.check_git_lfs_installed():
        setup.install_git_lfs_instructions()
        return
    
    # Configurar Git LFS
    if not setup.setup_git_lfs():
        print("❌ Error configurando Git LFS. Abortando.")
        return
    
    # Crear configuración
    setup.create_gitattributes()
    setup.create_shared_data_structure()
    setup.create_github_workflows()
    setup.create_data_sharing_config()
    
    # Mostrar próximos pasos
    setup.show_next_steps()

if __name__ == "__main__":
    main()
