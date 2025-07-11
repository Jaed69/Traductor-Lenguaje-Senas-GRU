"""
MediaPipe Models Auto-Downloader
Descarga automática de modelos requeridos para el sistema LSP
"""

import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
import time


class MediaPipeModelDownloader:
    """
    Gestor de descarga automática de modelos MediaPipe
    Descarga y verifica la integridad de los modelos necesarios
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuración de modelos requeridos
        self.required_models = {
            'hand_landmarker.task': {
                'url': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                'filename': 'hand_landmarker.task',
                'size_mb': 11.2,
                'description': 'Modelo de landmarks de manos',
                'sha256': None  # Se puede agregar para verificación de integridad
            },
            'pose_landmarker_heavy.task': {
                'url': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
                'filename': 'pose_landmarker_heavy.task',
                'size_mb': 12.8,
                'description': 'Modelo de landmarks de pose (pesado)',
                'sha256': None
            }
        }
    
    def check_models_availability(self) -> Dict[str, bool]:
        """
        Verifica qué modelos están disponibles localmente
        
        Returns:
            Dict con el estado de cada modelo requerido
        """
        status = {}
        
        for model_name, config in self.required_models.items():
            model_path = self.models_dir / config['filename']
            status[model_name] = model_path.exists() and model_path.stat().st_size > 1024  # > 1KB
        
        return status
    
    def download_model(self, model_name: str, show_progress: bool = True) -> bool:
        """
        Descarga un modelo específico
        
        Args:
            model_name: Nombre del modelo a descargar
            show_progress: Si mostrar progreso de descarga
            
        Returns:
            True si la descarga fue exitosa
        """
        if model_name not in self.required_models:
            print(f"❌ Modelo desconocido: {model_name}")
            return False
        
        config = self.required_models[model_name]
        model_path = self.models_dir / config['filename']
        
        # Si ya existe y es válido, no descargar
        if model_path.exists() and model_path.stat().st_size > 1024:
            if show_progress:
                print(f"✅ {config['description']} ya está disponible")
            return True
        
        if show_progress:
            print(f"📥 Descargando {config['description']}...")
            print(f"   📊 Tamaño: ~{config['size_mb']} MB")
            print(f"   🔗 URL: {config['url']}")
        
        try:
            # Realizar descarga con progreso
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Mostrar progreso
                        if show_progress and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r   📊 Descargando: {progress:.1f}%", end='', flush=True)
            
            if show_progress:
                print(f"\n✅ {config['description']} descargado exitosamente")
                print(f"   📁 Guardado en: {model_path}")
            
            # Verificar tamaño del archivo descargado
            if model_path.stat().st_size < 1024:
                print(f"❌ Error: Archivo descargado muy pequeño")
                model_path.unlink()  # Eliminar archivo corrupto
                return False
            
            return True
            
        except requests.RequestException as e:
            print(f"❌ Error descargando {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Limpiar archivo parcial
            return False
        except Exception as e:
            print(f"❌ Error inesperado descargando {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()
            return False
    
    def download_all_models(self, force_redownload: bool = False) -> bool:
        """
        Descarga todos los modelos requeridos
        
        Args:
            force_redownload: Si forzar re-descarga de modelos existentes
            
        Returns:
            True si todos los modelos se descargaron exitosamente
        """
        print("🔄 VERIFICANDO MODELOS MEDIAPIPE")
        print("="*50)
        
        # Verificar estado actual
        status = self.check_models_availability()
        missing_models = [name for name, available in status.items() if not available or force_redownload]
        
        if not missing_models and not force_redownload:
            print("✅ Todos los modelos MediaPipe están disponibles")
            return True
        
        print(f"📥 Necesario descargar {len(missing_models)} modelo(s)")
        
        # Estimar tiempo y tamaño total
        total_size_mb = sum(self.required_models[name]['size_mb'] for name in missing_models)
        estimated_time_min = max(1, total_size_mb / 5)  # ~5MB/min estimado
        
        print(f"📊 Tamaño total: ~{total_size_mb:.1f} MB")
        print(f"⏱️ Tiempo estimado: ~{estimated_time_min:.1f} minutos")
        
        # Preguntar confirmación para descargas grandes
        if total_size_mb > 20:
            confirm = input("\n¿Continuar con la descarga? (s/n): ").strip().lower()
            if confirm not in ['s', 'si', 'y', 'yes', '']:
                print("❌ Descarga cancelada por el usuario")
                return False
        
        print("\n🚀 Iniciando descarga de modelos...")
        start_time = time.time()
        
        # Descargar cada modelo
        success_count = 0
        for i, model_name in enumerate(missing_models, 1):
            print(f"\n📦 Modelo {i}/{len(missing_models)}: {model_name}")
            if self.download_model(model_name, show_progress=True):
                success_count += 1
            else:
                print(f"❌ Falló descarga de {model_name}")
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"📊 RESUMEN DESCARGA:")
        print(f"   ✅ Exitosos: {success_count}/{len(missing_models)}")
        print(f"   ⏱️ Tiempo total: {elapsed_time:.1f}s")
        
        if success_count == len(missing_models):
            print("🎉 ¡Todos los modelos descargados exitosamente!")
            print("✅ Sistema listo para usar MediaPipe")
            return True
        else:
            print("⚠️ Algunos modelos no se pudieron descargar")
            print("💡 Verifica tu conexión a internet e intenta nuevamente")
            return False
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """
        Verifica la integridad de un modelo descargado
        
        Args:
            model_name: Nombre del modelo a verificar
            
        Returns:
            True si el modelo es válido
        """
        if model_name not in self.required_models:
            return False
        
        config = self.required_models[model_name]
        model_path = self.models_dir / config['filename']
        
        if not model_path.exists():
            return False
        
        # Verificar tamaño mínimo
        file_size = model_path.stat().st_size
        if file_size < 1024 * 1024:  # Menos de 1MB es sospechoso
            return False
        
        # TODO: Verificar SHA256 si está disponible
        if config.get('sha256'):
            with open(model_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                return file_hash == config['sha256']
        
        return True
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Obtiene la ruta de un modelo específico
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Path del modelo si existe, None si no
        """
        if model_name not in self.required_models:
            return None
        
        config = self.required_models[model_name]
        model_path = self.models_dir / config['filename']
        
        return model_path if model_path.exists() else None
    
    def cleanup_invalid_models(self) -> int:
        """
        Limpia modelos corruptos o inválidos
        
        Returns:
            Número de modelos limpiados
        """
        cleaned = 0
        
        for model_name in self.required_models:
            if not self.verify_model_integrity(model_name):
                config = self.required_models[model_name]
                model_path = self.models_dir / config['filename']
                
                if model_path.exists():
                    print(f"🗑️ Eliminando modelo corrupto: {model_name}")
                    model_path.unlink()
                    cleaned += 1
        
        return cleaned
    
    def get_download_status(self) -> Dict[str, dict]:
        """
        Obtiene estado detallado de todos los modelos
        
        Returns:
            Dict con información detallada de cada modelo
        """
        status = {}
        
        for model_name, config in self.required_models.items():
            model_path = self.models_dir / config['filename']
            
            if model_path.exists():
                file_size = model_path.stat().st_size
                size_mb = file_size / (1024 * 1024)
                is_valid = self.verify_model_integrity(model_name)
            else:
                size_mb = 0
                is_valid = False
            
            status[model_name] = {
                'available': model_path.exists(),
                'valid': is_valid,
                'size_mb': size_mb,
                'expected_size_mb': config['size_mb'],
                'description': config['description'],
                'path': str(model_path) if model_path.exists() else None
            }
        
        return status


def setup_mediapipe_models(models_dir: str = "models", auto_download: bool = True) -> bool:
    """
    Función de conveniencia para configurar modelos MediaPipe
    
    Args:
        models_dir: Directorio donde guardar los modelos
        auto_download: Si descargar automáticamente modelos faltantes
        
    Returns:
        True si todos los modelos están disponibles
    """
    downloader = MediaPipeModelDownloader(models_dir)
    
    # Verificar modelos existentes
    status = downloader.check_models_availability()
    missing = [name for name, available in status.items() if not available]
    
    if not missing:
        print("✅ Todos los modelos MediaPipe están disponibles")
        return True
    
    if not auto_download:
        print(f"❌ Faltan {len(missing)} modelos MediaPipe")
        for model_name in missing:
            config = downloader.required_models[model_name]
            print(f"   • {config['description']}")
        return False
    
    # Descargar modelos faltantes
    return downloader.download_all_models()


if __name__ == "__main__":
    # Test del downloader
    downloader = MediaPipeModelDownloader()
    success = downloader.download_all_models()
    print(f"\n{'✅' if success else '❌'} Setup MediaPipe: {'Exitoso' if success else 'Falló'}")
