"""
Test específico para el módulo de recolección de datos
Tests más detallados para verificar funcionalidad del collector

Autor: LSP Team
Versión: 2.0 - Julio 2025
"""

import sys
import os
import unittest

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataCollector(unittest.TestCase):
    """Tests para el módulo de recolección de datos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        try:
            from src.data_collection.main_collector import LSPDataCollector
            self.collector_class = LSPDataCollector
        except ImportError as e:
            self.skipTest(f"No se pudo importar LSPDataCollector: {e}")
    
    def test_collector_initialization(self):
        """Test de inicialización del recolector"""
        try:
            collector = self.collector_class()
            self.assertIsNotNone(collector)
            self.assertEqual(collector.sequence_length, 60)
            self.assertEqual(collector.num_sequences, 50)
        except Exception as e:
            self.fail(f"Error inicializando recolector: {e}")
    
    def test_collector_modules(self):
        """Test de inicialización de módulos internos"""
        try:
            collector = self.collector_class()
            
            # Verificar que los módulos están inicializados
            self.assertIsNotNone(collector.mediapipe_manager)
            self.assertIsNotNone(collector.feature_extractor)
            self.assertIsNotNone(collector.motion_analyzer)
            self.assertIsNotNone(collector.ui_manager)
            self.assertIsNotNone(collector.data_manager)
            self.assertIsNotNone(collector.sign_config)
            
        except Exception as e:
            self.fail(f"Error verificando módulos: {e}")
    
    def test_signs_configuration(self):
        """Test de configuración de señas"""
        try:
            collector = self.collector_class()
            signs = collector.signs_to_collect
            
            self.assertIsInstance(signs, list)
            self.assertGreater(len(signs), 0)
            
        except Exception as e:
            self.fail(f"Error verificando configuración de señas: {e}")

def run_collector_tests():
    """Ejecuta tests específicos del collector"""
    print("🧪 TESTS DEL MÓDULO DE RECOLECCIÓN")
    print("="*45)
    
    # Crear suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataCollector)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Mostrar resumen
    print(f"\n📋 RESUMEN:")
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    
    success = len(result.errors) == 0 and len(result.failures) == 0
    
    if success:
        print("✅ Todos los tests del collector pasaron")
    else:
        print("❌ Algunos tests fallaron")
        
        if result.errors:
            print("\n🔴 ERRORES:")
            for test, error in result.errors:
                print(f"   {test}: {error}")
        
        if result.failures:
            print("\n🟡 FALLOS:")
            for test, failure in result.failures:
                print(f"   {test}: {failure}")
    
    return success

if __name__ == "__main__":
    run_collector_tests()
