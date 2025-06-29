# create_data_release.py
"""
Script para crear releases de datasets en GitHub.
Comprime y organiza los datos para distribución fácil.
"""

import os
import json
import zipfile
import shutil
import argparse
from datetime import datetime

class DataReleaseCreator:
    def __init__(self, version, description=""):
        self.version = version
        self.description = description
        self.release_dir = f"releases/{version}"
        self.temp_dir = f"temp_release_{version}"
        
    def analyze_current_dataset(self):
        """Analiza el dataset actual para incluir en metadatos"""
        stats = {
            "version": self.version,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": self.description,
            "total_sequences": 0,
            "total_signs": 0,
            "contributors": [],
            "signs_distribution": {},
            "quality_metrics": {}
        }
        
        # Analizar shared_data/contributors
        contributors_path = "shared_data/contributors"
        if os.path.exists(contributors_path):
            for contributor in os.listdir(contributors_path):
                contributor_path = os.path.join(contributors_path, contributor)
                if os.path.isdir(contributor_path):
                    stats["contributors"].append(contributor)
        
        # Analizar data/sequences local
        sequences_path = "data/sequences"
        if os.path.exists(sequences_path):
            for sign in os.listdir(sequences_path):
                sign_path = os.path.join(sequences_path, sign)
                if os.path.isdir(sign_path):
                    sequence_count = len([f for f in os.listdir(sign_path) if f.endswith('.npy')])
                    stats["signs_distribution"][sign] = sequence_count
                    stats["total_sequences"] += sequence_count
            
            stats["total_signs"] = len(stats["signs_distribution"])
        
        return stats
    
    def create_merged_dataset(self):
        """Crea un dataset combinado con todos los datos disponibles"""
        print("🔄 Creando dataset combinado...")
        
        merged_path = os.path.join(self.temp_dir, "merged_dataset")
        os.makedirs(merged_path, exist_ok=True)
        
        # Copiar datos locales
        local_sequences = "data/sequences"
        if os.path.exists(local_sequences):
            for sign in os.listdir(local_sequences):
                sign_path = os.path.join(local_sequences, sign)
                if os.path.isdir(sign_path):
                    target_sign_path = os.path.join(merged_path, sign)
                    os.makedirs(target_sign_path, exist_ok=True)
                    
                    # Copiar secuencias numerando desde 0
                    sequence_counter = 0
                    for seq_file in os.listdir(sign_path):
                        if seq_file.endswith('.npy'):
                            src = os.path.join(sign_path, seq_file)
                            dst = os.path.join(target_sign_path, f"{sequence_counter}.npy")
                            shutil.copy2(src, dst)
                            sequence_counter += 1
        
        # Agregar datos de contribuidores
        contributors_path = "shared_data/contributors"
        if os.path.exists(contributors_path):
            for contributor in os.listdir(contributors_path):
                contributor_sequences = os.path.join(contributors_path, contributor, "sequences")
                if os.path.exists(contributor_sequences):
                    for sign in os.listdir(contributor_sequences):
                        sign_path = os.path.join(contributor_sequences, sign)
                        if os.path.isdir(sign_path):
                            target_sign_path = os.path.join(merged_path, sign)
                            os.makedirs(target_sign_path, exist_ok=True)
                            
                            # Continuar numeración desde donde quedó
                            existing_files = [f for f in os.listdir(target_sign_path) if f.endswith('.npy')]
                            sequence_counter = len(existing_files)
                            
                            for seq_file in os.listdir(sign_path):
                                if seq_file.endswith('.npy'):
                                    src = os.path.join(sign_path, seq_file)
                                    dst = os.path.join(target_sign_path, f"{sequence_counter}.npy")
                                    shutil.copy2(src, dst)
                                    sequence_counter += 1
        
        print(f"✅ Dataset combinado creado en {merged_path}")
        return merged_path
    
    def create_sample_dataset(self):
        """Crea un dataset pequeño de muestra"""
        print("🔄 Creando dataset de muestra...")
        
        sample_path = os.path.join(self.temp_dir, "sample_dataset")
        merged_path = os.path.join(self.temp_dir, "merged_dataset")
        
        if not os.path.exists(merged_path):
            print("❌ Dataset combinado no encontrado")
            return None
        
        os.makedirs(sample_path, exist_ok=True)
        
        # Tomar solo 5 secuencias de cada seña para la muestra
        for sign in os.listdir(merged_path):
            sign_path = os.path.join(merged_path, sign)
            if os.path.isdir(sign_path):
                sample_sign_path = os.path.join(sample_path, sign)
                os.makedirs(sample_sign_path, exist_ok=True)
                
                sequences = [f for f in os.listdir(sign_path) if f.endswith('.npy')][:5]
                for i, seq_file in enumerate(sequences):
                    src = os.path.join(sign_path, seq_file)
                    dst = os.path.join(sample_sign_path, f"{i}.npy")
                    shutil.copy2(src, dst)
        
        print(f"✅ Dataset de muestra creado en {sample_path}")
        return sample_path
    
    def create_documentation(self, stats):
        """Crea documentación para el release"""
        readme_content = f"""# Dataset Release {self.version}

{self.description}

## 📊 Estadísticas del Dataset

- **Versión**: {stats['version']}
- **Fecha de creación**: {stats['creation_date']}
- **Total de señas**: {stats['total_signs']}
- **Total de secuencias**: {stats['total_sequences']}
- **Contribuidores**: {len(stats['contributors'])}

## 📋 Distribución por Señas

| Seña | Secuencias |
|------|------------|
"""
        
        for sign, count in sorted(stats['signs_distribution'].items()):
            readme_content += f"| {sign} | {count} |\n"
        
        readme_content += f"""
## 👥 Contribuidores

{', '.join(stats['contributors']) if stats['contributors'] else 'Datos locales únicamente'}

## 📁 Contenido del Release

- `merged_dataset/` - Dataset completo combinado
- `sample_dataset/` - Dataset pequeño para pruebas (5 secuencias por seña)
- `metadata.json` - Metadatos detallados del dataset
- `README.md` - Esta documentación

## 🚀 Uso

### Descargar y usar el dataset completo:
```bash
# Descargar release
wget https://github.com/TU_USUARIO/MediaLengS/releases/download/{self.version}/dataset_{self.version}.zip

# Extraer
unzip dataset_{self.version}.zip

# Copiar al proyecto
cp -r merged_dataset/* data/sequences/

# Entrenar modelo
python model_trainer_sequence.py
```

### Solo dataset de muestra:
```bash
# Usar sample_dataset para pruebas rápidas
cp -r sample_dataset/* data/sequences/
python model_trainer_sequence.py --epochs 10
```

## 📈 Mejoras en esta Versión

{self.description}

## 🔄 Próxima Versión

Para contribuir datos para la próxima versión:
1. Fork del repositorio
2. Agregar datos en `shared_data/contributors/`
3. Pull request con tus contribuciones
"""
        
        readme_path = os.path.join(self.temp_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Crear metadata.json
        metadata_path = os.path.join(self.temp_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print("✅ Documentación creada")
    
    def create_zip_release(self):
        """Crea archivo ZIP del release"""
        print("🔄 Creando archivo ZIP del release...")
        
        zip_name = f"dataset_{self.version}.zip"
        zip_path = os.path.join("releases", zip_name)
        
        os.makedirs("releases", exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zipf.write(file_path, arcname)
        
        # Obtener tamaño del archivo
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        
        print(f"✅ Release ZIP creado: {zip_path}")
        print(f"   Tamaño: {size_mb:.1f} MB")
        
        return zip_path, size_mb
    
    def cleanup_temp_files(self):
        """Limpia archivos temporales"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Archivos temporales limpiados")
    
    def create_release(self):
        """Crea el release completo"""
        print(f"🚀 Creando release {self.version}...")
        print("="*50)
        
        try:
            # Crear directorio temporal
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Analizar dataset actual
            stats = self.analyze_current_dataset()
            print(f"📊 Dataset analizado: {stats['total_signs']} señas, {stats['total_sequences']} secuencias")
            
            # Crear datasets
            merged_path = self.create_merged_dataset()
            sample_path = self.create_sample_dataset()
            
            # Crear documentación
            self.create_documentation(stats)
            
            # Crear ZIP
            zip_path, size_mb = self.create_zip_release()
            
            # Limpiar archivos temporales
            self.cleanup_temp_files()
            
            print("\n" + "="*50)
            print("🎉 ¡RELEASE CREADO EXITOSAMENTE!")
            print("="*50)
            print(f"📦 Archivo: {zip_path}")
            print(f"📏 Tamaño: {size_mb:.1f} MB")
            print(f"📊 Contenido: {stats['total_signs']} señas, {stats['total_sequences']} secuencias")
            
            print("\n📋 PRÓXIMOS PASOS:")
            print("1. Subir a GitHub Release:")
            print(f"   - Ir a https://github.com/TU_USUARIO/MediaLengS/releases")
            print(f"   - Crear nueva release con tag '{self.version}'")
            print(f"   - Subir el archivo {zip_path} como asset")
            
            print("\n2. O usar GitHub CLI:")
            print(f"   gh release create {self.version} {zip_path} --title 'Dataset {self.version}' --notes '{self.description}'")
            
            return zip_path, stats
            
        except Exception as e:
            print(f"❌ Error creando release: {e}")
            self.cleanup_temp_files()
            return None, None

def main():
    parser = argparse.ArgumentParser(description='Crear release de dataset para GitHub')
    parser.add_argument('--version', required=True, help='Versión del release (ej: v1.0.0)')
    parser.add_argument('--description', default='', help='Descripción del release')
    parser.add_argument('--auto-upload', action='store_true', help='Subir automáticamente a GitHub (requiere gh CLI)')
    
    args = parser.parse_args()
    
    print("📦 CREADOR DE RELEASES DE DATASET")
    print("Traductor de Lenguaje de Señas - GitHub Data Release")
    print("="*60)
    
    creator = DataReleaseCreator(args.version, args.description)
    zip_path, stats = creator.create_release()
    
    if zip_path and args.auto_upload:
        print("\n🚀 Subiendo a GitHub...")
        try:
            import subprocess
            result = subprocess.run([
                'gh', 'release', 'create', args.version, zip_path,
                '--title', f'Dataset {args.version}',
                '--notes', args.description or f'Dataset release {args.version}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Release subido exitosamente a GitHub")
            else:
                print(f"❌ Error subiendo a GitHub: {result.stderr}")
        except FileNotFoundError:
            print("❌ GitHub CLI (gh) no está instalado. Sube manualmente el archivo.")

if __name__ == "__main__":
    main()
