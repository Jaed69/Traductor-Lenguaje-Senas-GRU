"""
Test Suite for LSP System
Tests básicos para verificar funcionamiento de módulos

Autor: LSP Team
Versión: 2.0 - Julio 2025
"""

import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_data_collection():
    """Test básico del módulo de recolección de datos"""
    try:
        from src.data_collection.main_collector import LSPDataCollector
        print("✅ Módulo de recolección de datos: OK")
        
        # Test de inicialización
        collector = LSPDataCollector()
        print("✅ Inicialización del recolector: OK")
        
        return True
    except Exception as e:
        print(f"❌ Error en módulo de recolección: {e}")
        return False

def test_training():
    """Test básico del módulo de entrenamiento"""
    try:
        from src.training.train_gru import GRUTrainer
        print("✅ Módulo de entrenamiento: OK")
        
        # Test de inicialización
        trainer = GRUTrainer()
        print("✅ Inicialización del entrenador: OK")
        
        return True
    except Exception as e:
        print(f"❌ Error en módulo de entrenamiento: {e}")
        return False

def test_evaluation():
    """Test básico del módulo de evaluación"""
    try:
        from src.evaluation.evaluate_model import ModelEvaluator
        print("✅ Módulo de evaluación: OK")
        
        # Test de inicialización
        evaluator = ModelEvaluator()
        print("✅ Inicialización del evaluador: OK")
        
        return True
    except Exception as e:
        print(f"❌ Error en módulo de evaluación: {e}")
        return False

def test_inference():
    """Test básico del módulo de inferencia"""
    try:
        from src.inference.real_time_translator import RealTimeTranslator
        print("✅ Módulo de inferencia: OK")
        
        # Test de inicialización
        translator = RealTimeTranslator()
        print("✅ Inicialización del traductor: OK")
        
        return True
    except Exception as e:
        print(f"❌ Error en módulo de inferencia: {e}")
        return False

def test_dependencies():
    """Test de dependencias principales"""
    dependencies = [
        'cv2', 'mediapipe', 'numpy', 'collections',
        'json', 'os', 'time', 'datetime'
    ]
    
    failed_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: Disponible")
        except ImportError:
            print(f"❌ {dep}: No disponible")
            failed_deps.append(dep)
    
    return len(failed_deps) == 0

def run_all_tests():
    """Ejecuta todos los tests"""
    print("🧪 EJECUTANDO TESTS DEL SISTEMA LSP")
    print("="*50)
    
    # Test de dependencias
    print("\n📦 Verificando dependencias...")
    deps_ok = test_dependencies()
    
    # Test de módulos
    print("\n🧩 Verificando módulos...")
    collection_ok = test_data_collection()
    training_ok = test_training()
    evaluation_ok = test_evaluation()
    inference_ok = test_inference()
    
    # Resumen
    print("\n📋 RESUMEN DE TESTS")
    print("="*30)
    print(f"Dependencias: {'✅' if deps_ok else '❌'}")
    print(f"Recolección:  {'✅' if collection_ok else '❌'}")
    print(f"Entrenamiento: {'✅' if training_ok else '❌'}")
    print(f"Evaluación:   {'✅' if evaluation_ok else '❌'}")
    print(f"Inferencia:   {'✅' if inference_ok else '❌'}")
    
    all_ok = all([deps_ok, collection_ok, training_ok, evaluation_ok, inference_ok])
    
    if all_ok:
        print("\n🎉 TODOS LOS TESTS PASARON")
    else:
        print("\n⚠️ ALGUNOS TESTS FALLARON")
        print("💡 Revisa las dependencias y la configuración")
    
    return all_ok

if __name__ == "__main__":
    run_all_tests()
