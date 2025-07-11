"""
Test rápido del nuevo menú con indicadores de progreso
Versión: 2.1 - Julio 2025
"""
import sys
import os
import pytest

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_improved_menu():
    """Prueba el menú mejorado, con fallback si MediaPipe no está disponible."""
    try:
        from src.data_collection.main_collector import LSPDataCollector
        collector = LSPDataCollector()
        
        test_signs = collector.signs_to_collect[:5]
        
        # Probar que las funciones de UI no lanzan excepciones con el colector completo
        collector.ui_manager.show_menu(
            test_signs, 
            collector.data_manager, 
            collector.sign_config
        )
        collector.ui_manager.show_detailed_statistics(
            test_signs,
            collector.data_manager, 
            collector.sign_config
        )
    except RuntimeError as e:
        if 'MediaPipe' in str(e):
            # Si falla por MediaPipe, probar con managers independientes como fallback
            try:
                from src.data_collection.data_manager import DataManager
                from src.data_collection.sign_config import SignConfig
                from src.data_collection.ui_manager import UIManager
                
                dm = DataManager()
                sc = SignConfig()
                ui = UIManager()
                test_signs = ["A", "B", "C", "HOLA", "GRACIAS"]
                
                ui.show_menu(test_signs, dm, sc)
                ui.show_detailed_statistics(test_signs, dm, sc)
            except Exception as e2:
                pytest.fail(f"El test de fallback (sin MediaPipe) falló: {e2}")
        else:
            # Si es otro RuntimeError, la prueba debe fallar
            pytest.fail(f"El test falló con un error inesperado de Runtime: {e}")
    except Exception as e:
        pytest.fail(f"El test falló con una excepción inesperada: {e}")