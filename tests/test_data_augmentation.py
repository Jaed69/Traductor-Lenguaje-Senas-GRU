"""
Test completo del sistema de Data Augmentation para LSP
Verifica todas las funcionalidades del módulo de augmentación
Versión: 2.1 - Julio 2025
"""

import sys
import os
import numpy as np
import pytest

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar clases necesarias
from src.data_collection.data_augmentation import LSPDataAugmenter, AugmentationIntegrator
from src.data_collection.data_manager import DataManager
from src.data_collection.sign_config import SignConfig
from src.data_collection.main_collector import LSPDataCollector
from src.data_collection.ui_manager import UIManager

@pytest.fixture
def augmenter():
    """Fixture para instanciar el LSPDataAugmenter."""
    return LSPDataAugmenter()

@pytest.fixture
def test_sequence():
    """Fixture para crear una secuencia de prueba estándar."""
    return np.random.rand(60, 157).astype(np.float32)

def test_data_augmenter_initialization(augmenter):
    """Prueba que el LSPDataAugmenter se inicializa correctamente."""
    assert augmenter is not None

def test_sequence_augmentation(augmenter, test_sequence):
    """Prueba la generación de múltiples aumentos para una secuencia."""
    metadata = {'sign': 'TEST', 'sign_type': 'word'}
    num_augmentations = 3
    augmented_data = augmenter.augment_sequence(test_sequence, 'word', metadata, num_augmentations)
    
    assert len(augmented_data) == num_augmentations
    for seq, meta in augmented_data:
        assert seq.shape == test_sequence.shape
        assert 'augmentation' in meta

@pytest.mark.parametrize("technique", [
    'temporal_light', 'spatial_light', 'noise_light', 'hand_variations'
])
def test_specific_augmentation_techniques(augmenter, test_sequence, technique):
    """Prueba que cada técnica de aumentación individual funciona."""
    try:
        augmented_seq = augmenter._apply_augmentation(test_sequence, technique)
        assert augmented_seq.shape == test_sequence.shape
        assert np.all((augmented_seq >= 0) & (augmented_seq <= 1)), "Los valores deben estar en el rango [0, 1]"
    except Exception as e:
        pytest.fail(f"La técnica de aumentación '{technique}' falló con una excepción: {e}")

@pytest.fixture
def integrator():
    """Fixture para el AugmentationIntegrator."""
    # Usar un directorio de datos de prueba si es posible para aislar los tests
    dm = DataManager(data_dir="tests/test_data") 
    sc = SignConfig()
    return AugmentationIntegrator(dm, sc)

def test_augmentation_needs_calculation(integrator):
    """Prueba el cálculo de necesidades de aumentación."""
    current = {'HOLA': 10, 'GRACIAS': 5, 'A': 15}
    target = {'HOLA': 50, 'GRACIAS': 50, 'A': 30}
    needs = integrator.augmenter.calculate_augmentation_needs(current, target)
    
    assert needs['HOLA'] == 30
    assert needs['GRACIAS'] == 34
    assert needs['A'] == 12

def test_main_collector_has_augmentation_integrator():
    """Verifica que el colector principal tenga una instancia del integrador."""
    try:
        collector = LSPDataCollector()
        assert hasattr(collector, 'augmentation_integrator')
        assert hasattr(collector, '_run_data_augmentation')
    except Exception as e:
        # Puede fallar si MediaPipe no está disponible, pero la estructura debe ser correcta
        if "MediaPipe" in str(e):
            pytest.skip("MediaPipe no disponible, saltando test de integración del colector.")
        else:
            pytest.fail(f"Falló la inicialización del colector: {e}")

def test_ui_manager_has_augmentation_menu():
    """Verifica que el UIManager tenga los métodos del menú de aumentación."""
    ui = UIManager()
    assert hasattr(ui, 'show_augmentation_menu')
    assert hasattr(ui, 'get_augmentation_choice')

# El bloque main se puede mantener para demostraciones manuales
def main():
    """Ejecuta una demostración del flujo de aumentación."""
    print("🚀 DEMO DEL SISTEMA DE DATA AUGMENTATION")
    print("="*70)
    
    # Preparar entorno de demo
    test_data_dir = "tests/test_data_demo"
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
        
    dm = DataManager(data_dir=test_data_dir)
    sc = SignConfig()
    integrator = AugmentationIntegrator(dm, sc)
    
    # Simular algunas secuencias existentes
    print("1. Simulando datos existentes...")
    for sign in ["HOLA", "ADIOS"]:
        for i in range(5):
            seq = np.random.rand(60, 157).astype(np.float32)
            meta = dm.create_metadata(sign, sc.classify_sign_type(sign), {}, 90, "EXCELENTE", [], [], "DEMO")
            dm.save_sequence(seq, sign, i + 1, meta)
    print("   Datos simulados creados.")

    # Ejecutar aumentación
    print("\n2. Ejecutando proceso de aumentación automática...")
    try:
        report = integrator.auto_augment_dataset(target_count=10)
        print("   ¡Proceso completado!")
        print(f"   - Señas procesadas: {report['signs_processed']}")
        print(f"   - Secuencias originales: {report['total_original']}")
        print(f"   - Secuencias aumentadas: {report['total_augmented']}")
    except Exception as e:
        print(f"   Ocurrió un error durante la aumentación: {e}")

    # Limpiar
    print("\n3. Limpiando datos de la demo...")
    # shutil.rmtree(test_data_dir) # Descomentar para limpiar automáticamente
    print(f"   El directorio '{test_data_dir}' puede ser eliminado manualmente.")

if __name__ == "__main__":
    main()