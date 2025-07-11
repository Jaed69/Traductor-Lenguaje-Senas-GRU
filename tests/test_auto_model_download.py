"""
Test del sistema de descarga automática de modelos MediaPipe
Versión: 2.1 - Julio 2025
"""

import os
import sys
import pytest

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar las clases necesarias una sola vez
from src.utils.mediapipe_model_downloader import MediaPipeModelDownloader, setup_mediapipe_models

@pytest.fixture
def downloader():
    """Fixture de Pytest para inicializar el downloader en un directorio temporal."""
    # Idealmente, usar un directorio de prueba temporal, pero por simplicidad usamos 'models'
    return MediaPipeModelDownloader(models_dir="models")

def test_model_downloader_initialization(downloader):
    """Prueba que el descargador de modelos se inicializa correctamente."""
    assert downloader is not None, "El downloader no debería ser None"
    assert len(downloader.required_models) > 0, "Deberían existir modelos requeridos"

def test_model_availability_check(downloader):
    """Prueba la verificación de disponibilidad de modelos."""
    try:
        status = downloader.check_models_availability()
        assert isinstance(status, dict), "El estado debe ser un diccionario"
        # La disponibilidad real depende del entorno, así que solo verificamos que no falle
    except Exception as e:
        pytest.fail(f"La verificación de disponibilidad falló con una excepción: {e}")

def test_detailed_status_check(downloader):
    """Prueba la obtención del estado detallado de los modelos."""
    try:
        detailed_status = downloader.get_download_status()
        assert isinstance(detailed_status, dict), "El estado detallado debe ser un diccionario"
        for model_name, info in detailed_status.items():
            assert 'available' in info
            assert 'valid' in info
            assert 'size_mb' in info
    except Exception as e:
        pytest.fail(f"La obtención de estado detallado falló con una excepción: {e}")

def test_setup_function_verification_mode():
    """Prueba la función setup en modo de solo verificación."""
    try:
        # Solo verificar, no descargar automáticamente en el test
        models_ok = setup_mediapipe_models(auto_download=False)
        # El resultado puede ser True o False, pero no debe lanzar una excepción
        assert isinstance(models_ok, bool), "La función setup debe devolver un booleano"
    except Exception as e:
        pytest.fail(f"setup_mediapipe_models falló en modo verificación: {e}")

def test_integration_with_main_system():
    """Prueba la integración simulada con el sistema principal."""
    try:
        # Simular la verificación de modelos que haría run.py
        models_status = setup_mediapipe_models(auto_download=False)
        assert isinstance(models_status, bool), "La verificación de integración debe devolver un booleano"
    except Exception as e:
        pytest.fail(f"La integración con el sistema principal falló: {e}")

# El bloque main se mantiene para la ejecución manual y demostrativa
def main():
    """Ejecuta una demostración del proceso de verificación."""
    print("🚀 DEMO DEL SISTEMA DE DESCARGA AUTOMÁTICA MEDIAPIPE")
    print("="*70)
    
    try:
        downloader = MediaPipeModelDownloader("models")
        print("✅ MediaPipeModelDownloader inicializado")
        
        detailed_status = downloader.get_download_status()
        print("\n📋 Estado detallado de los modelos:")
        for model_name, info in detailed_status.items():
            status_icon = "✅" if info['available'] and info['valid'] else "❌"
            print(f"   {status_icon} {model_name}: {'Disponible y válido' if info['available'] and info['valid'] else 'Faltante o inválido'}")
            print(f"      • Ruta: {info['path'] if info['path'] else 'N/A'}")

        print("\n🔧 Ejecutando setup_mediapipe_models(auto_download=False)... ")
        models_ok = setup_mediapipe_models(auto_download=False)
        
        if models_ok:
            print("\n🎉 ¡TODOS LOS MODELOS ESTÁN LISTOS!")
        else:
            print("\n⚠️ ALGUNOS MODELOS FALTAN O SON INVÁLIDOS.")
            print("💡 Al ejecutar 'python run.py', se intentará la descarga automática.")

    except Exception as e:
        print(f"\n🔥 Ocurrió un error durante la demostración: {e}")

if __name__ == "__main__":
    main()