# download_shared_data.py
"""
Script para descargar datos compartidos desde GitHub releases.
Facilita la obtención de datasets para nuevos colaboradores.
"""

import os
import json
import time
import shutil
import requests
import zipfile
import argparse
from urllib.parse import urlparse

class GitHubDataDownloader:
    def __init__(self, repo_url, version=None):
        self.repo_url = repo_url.rstrip('/')
        self.version = version
        self.api_base = self._get_api_url()
        
    def _get_api_url(self):
        """Convierte URL del repo a API URL"""
        if 'github.com' in self.repo_url:
            # Extraer owner/repo de la URL
            parts = self.repo_url.split('/')
            if len(parts) >= 2:
                owner = parts[-2]
                repo = parts[-1]
                return f"https://api.github.com/repos/{owner}/{repo}"
        return None
    
    def list_available_releases(self):
        """Lista todas las releases disponibles"""
        if not self.api_base:
            print("❌ URL de repositorio inválida")
            return []
        
        try:
            response = requests.get(f"{self.api_base}/releases")
            response.raise_for_status()
            releases = response.json()
            
            print("📦 RELEASES DISPONIBLES:")
            print("-" * 40)
            
            for release in releases:
                tag = release['tag_name']
                name = release['name']
                published = release['published_at'][:10]
                assets_count = len(release['assets'])
                
                print(f"🏷️  {tag} - {name}")
                print(f"   📅 Publicado: {published}")
                print(f"   📎 Assets: {assets_count}")
                
                # Mostrar assets
                for asset in release['assets']:
                    size_mb = asset['size'] / (1024 * 1024)
                    print(f"      📁 {asset['name']} ({size_mb:.1f} MB)")
                print()
            
            return releases
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error obteniendo releases: {e}")
            return []
    
    def download_release(self, version=None, output_dir="downloaded_data"):
        """Descarga una release específica"""
        version = version or self.version
        if not version:
            print("❌ Debe especificar una versión")
            return False
        
        print(f"🔄 Descargando release {version}...")
        
        try:
            # Obtener información de la release
            response = requests.get(f"{self.api_base}/releases/tags/{version}")
            response.raise_for_status()
            release = response.json()
            
            # Encontrar asset del dataset
            dataset_asset = None
            for asset in release['assets']:
                if 'dataset' in asset['name'].lower() and asset['name'].endswith('.zip'):
                    dataset_asset = asset
                    break
            
            if not dataset_asset:
                print(f"❌ No se encontró dataset en la release {version}")
                return False
            
            # Descargar el archivo
            download_url = dataset_asset['browser_download_url']
            filename = dataset_asset['name']
            size_mb = dataset_asset['size'] / (1024 * 1024)
            
            print(f"📥 Descargando {filename} ({size_mb:.1f} MB)...")
            
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Archivo descargado: {file_path}")
            
            # Extraer ZIP
            extract_path = os.path.join(output_dir, f"dataset_{version}")
            self._extract_zip(file_path, extract_path)
            
            return extract_path
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error descargando release: {e}")
            return False
    
    def _extract_zip(self, zip_path, extract_path):
        """Extrae archivo ZIP"""
        print(f"📂 Extrayendo a {extract_path}...")
        
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_path)
        
        print("✅ Archivo extraído exitosamente")
        
        # Mostrar contenido
        self._show_extracted_content(extract_path)
    
    def _show_extracted_content(self, extract_path):
        """Muestra el contenido extraído"""
        print(f"\n📋 CONTENIDO EXTRAÍDO en {extract_path}:")
        print("-" * 50)
        
        for item in os.listdir(extract_path):
            item_path = os.path.join(extract_path, item)
            if os.path.isdir(item_path):
                subdir_count = len([x for x in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, x))])
                file_count = len([x for x in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, x))])
                print(f"📁 {item}/ ({subdir_count} dirs, {file_count} files)")
            else:
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"📄 {item} ({size_mb:.1f} MB)")
    
    def install_to_project(self, extract_path, target="data/sequences", backup=True):
        """Instala el dataset descargado en el proyecto"""
        print(f"\n🔧 Instalando dataset en {target}...")
        
        # Buscar merged_dataset en el contenido extraído
        merged_dataset_path = None
        for item in os.listdir(extract_path):
            if item == "merged_dataset":
                merged_dataset_path = os.path.join(extract_path, item)
                break
        
        if not merged_dataset_path:
            print("❌ No se encontró merged_dataset en el contenido extraído")
            return False
        
        # Hacer backup si es necesario
        if backup and os.path.exists(target):
            backup_path = f"{target}_backup_{int(time.time())}"
            print(f"💾 Creando backup en {backup_path}")
            shutil.move(target, backup_path)
        
        # Copiar nuevo dataset
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copytree(merged_dataset_path, target)
        
        # Mostrar estadísticas
        self._show_dataset_stats(target)
        
        print(f"✅ Dataset instalado en {target}")
        return True
    
    def _show_dataset_stats(self, dataset_path):
        """Muestra estadísticas del dataset instalado"""
        if not os.path.exists(dataset_path):
            return
        
        total_sequences = 0
        signs_count = 0
        
        for sign in os.listdir(dataset_path):
            sign_path = os.path.join(dataset_path, sign)
            if os.path.isdir(sign_path):
                signs_count += 1
                sequences = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
                total_sequences += len(sequences)
        
        print(f"\n📊 ESTADÍSTICAS DEL DATASET INSTALADO:")
        print(f"   Señas: {signs_count}")
        print(f"   Secuencias totales: {total_sequences}")
        print(f"   Promedio por seña: {total_sequences/signs_count:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Descargar datos compartidos desde GitHub')
    parser.add_argument('--repo', required=True, help='URL del repositorio de GitHub')
    parser.add_argument('--version', help='Versión específica a descargar')
    parser.add_argument('--list', action='store_true', help='Listar releases disponibles')
    parser.add_argument('--output', default='downloaded_data', help='Directorio de salida')
    parser.add_argument('--install', action='store_true', help='Instalar automáticamente en data/sequences')
    parser.add_argument('--no-backup', action='store_true', help='No crear backup al instalar')
    
    args = parser.parse_args()
    
    print("📥 DESCARGADOR DE DATOS DE GITHUB")
    print("Traductor de Lenguaje de Señas - GitHub Data Downloader")
    print("="*60)
    
    downloader = GitHubDataDownloader(args.repo, args.version)
    
    if args.list:
        releases = downloader.list_available_releases()
        if releases:
            print(f"\n💡 Para descargar una release específica:")
            print(f"python {__file__} --repo {args.repo} --version TAG_NAME")
        return
    
    if not args.version:
        print("❌ Debe especificar --version o usar --list para ver opciones")
        return
    
    # Descargar release
    extract_path = downloader.download_release(args.version, args.output)
    
    if extract_path and args.install:
        import shutil
        import time
        
        backup = not args.no_backup
        success = downloader.install_to_project(extract_path, backup=backup)
        
        if success:
            print("\n🎉 ¡Dataset descargado e instalado exitosamente!")
            print("\n📋 PRÓXIMOS PASOS:")
            print("1. Verificar datos: python dataset_stats.py")
            print("2. Entrenar modelo: python model_trainer_sequence.py")
            print("3. Probar traductor: python main.py")
        else:
            print("❌ Error durante la instalación")
    elif extract_path:
        print(f"\n✅ Dataset descargado en: {extract_path}")
        print("\n📋 Para instalar manualmente:")
        print(f"cp -r {extract_path}/merged_dataset/* data/sequences/")

if __name__ == "__main__":
    main()
