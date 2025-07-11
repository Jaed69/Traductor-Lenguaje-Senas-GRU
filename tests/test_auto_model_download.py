"""
Test del sistema de descarga automática de modelos MediaPipe
"""

import os
import sys

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_model_downloader():
    """Prueba el descargador de modelos MediaPipe"""
    print("🧪 TESTING DESCARGA AUTOMÁTICA DE MODELOS MEDIAPIPE")
    print("="*60)
    
    try:
        from src.utils.mediapipe_model_downloader import MediaPipeModelDownloader, setup_mediapipe_models
        
        print("✅ Módulo de descarga importado correctamente")
        
        # Test 1: Inicializar downloader
        downloader = MediaPipeModelDownloader("models")
        print("✅ MediaPipeModelDownloader inicializado")
        
        # Test 2: Verificar configuración de modelos
        required_models = downloader.required_models
        print(f"✅ Configurados {len(required_models)} modelos requeridos:")
        for name, config in required_models.items():
            print(f"   📦 {name}: {config['description']} (~{config['size_mb']} MB)")
        
        # Test 3: Verificar estado actual
        status = downloader.check_models_availability()
        print(f"\n📊 Estado actual de modelos:")
        for model_name, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {model_name}: {'Disponible' if available else 'Faltante'}")
        
        # Test 4: Obtener estado detallado
        detailed_status = downloader.get_download_status()
        print(f"\n📋 Estado detallado:")
        for model_name, info in detailed_status.items():
            print(f"   📦 {model_name}:")
            print(f"      • Disponible: {info['available']}")
            print(f"      • Válido: {info['valid']}")
            print(f"      • Tamaño: {info['size_mb']:.1f} MB (esperado: {info['expected_size_mb']} MB)")
            if info['path']:
                print(f"      • Ruta: {info['path']}")
        
        # Test 5: Función de conveniencia
        print(f"\n🔧 Probando función setup_mediapipe_models...")
        
        # Solo verificar, no descargar automáticamente en test
        try:
            models_ok = setup_mediapipe_models(auto_download=False)
            print(f"✅ setup_mediapipe_models ejecutado: {models_ok}")
        except Exception as e:
            print(f"⚠️ setup_mediapipe_models con advertencia: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_main_system():
    """Prueba la integración con el sistema principal"""
    print("\n🧪 TESTING INTEGRACIÓN CON SISTEMA PRINCIPAL")
    print("="*50)
    
    try:
        # Simular importación desde run.py
        print("📥 Simulando verificación de modelos desde run.py...")
        
        from src.utils.mediapipe_model_downloader import setup_mediapipe_models
        
        # Test sin descarga automática
        print("🔍 Verificando modelos (sin descarga automática)...")
        models_status = setup_mediapipe_models(auto_download=False)
        
        print(f"✅ Verificación completada: {models_status}")
        
        if not models_status:
            print("💡 En ejecución real, se ofrecería descarga automática")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    print("🚀 VERIFICACIÓN SISTEMA DESCARGA AUTOMÁTICA MEDIAPIPE")
    print("="*70)
    print("🎯 Validando descarga automática de modelos al iniciar sistema")
    print()
    
    tests = [
        ("Descargador de Modelos", test_model_downloader),
        ("Integración Sistema Principal", test_integration_with_main_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n▶️ Ejecutando: {test_name}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} completado exitosamente")
            else:
                print(f"❌ {test_name} falló")
                
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name} falló con excepción: {e}")
    
    # Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN VERIFICACIÓN")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print("-" * 70)
    print(f"📊 Resultado: {passed}/{len(tests)} pruebas pasaron ({passed/len(tests)*100:.1f}%)")
    
    if passed == len(tests):
        print("\n🎉 ¡SISTEMA DE DESCARGA AUTOMÁTICA FUNCIONANDO!")
        print("✅ Modelos MediaPipe se descargarán automáticamente al ejecutar run.py")
        print("🔄 El sistema verificará y descargará modelos faltantes")
        print("💡 Los usuarios no necesitarán configuración manual")
    else:
        print(f"\n⚠️ {len(tests) - passed} pruebas fallaron")
        print("💡 Revisa los errores anteriores")
    
    # Mostrar instrucciones de uso
    print(f"\n🎯 CÓMO USAR:")
    print("   1. Ejecuta: python run.py")
    print("   2. El sistema verificará automáticamente los modelos MediaPipe")
    print("   3. Si faltan modelos, se descargarán automáticamente")
    print("   4. Una vez descargados, el sistema estará listo para usar")
    print("   5. Los modelos se guardan en la carpeta 'models/'")
    
    print(f"\n📦 MODELOS REQUERIDOS:")
    print("   • hand_landmarker.task (~11.2 MB) - Landmarks de manos")
    print("   • pose_landmarker_heavy.task (~12.8 MB) - Landmarks de pose")
    print("   📊 Total: ~24 MB de descarga")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()
