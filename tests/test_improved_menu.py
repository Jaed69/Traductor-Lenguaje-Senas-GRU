"""
Test rápido del nuevo menú con indicadores de progreso
"""
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_improved_menu():
    """Prueba el menú mejorado con indicadores de progreso"""
    print("🧪 TESTING MENÚ MEJORADO CON PROGRESO")
    print("="*50)
    
    try:
        from src.data_collection.main_collector import LSPDataCollector
        print("✅ LSPDataCollector importado")
        
        # Intentar inicializar (puede fallar por MediaPipe)
        try:
            collector = LSPDataCollector()
            print("✅ LSPDataCollector inicializado")
            print(f"📊 Señas configuradas: {len(collector.signs_to_collect)}")
            
            # Probar el nuevo menú con managers
            print("\n📋 PROBANDO NUEVO MENÚ:")
            print("-" * 30)
            
            # Usar solo las primeras 5 señas para testing
            test_signs = collector.signs_to_collect[:5]
            collector.ui_manager.show_menu(
                test_signs, 
                collector.data_manager, 
                collector.sign_config
            )
            
            print("\n✅ Menú mejorado mostrado correctamente")
            
            # Probar estadísticas detalladas
            print("\n📊 PROBANDO ESTADÍSTICAS DETALLADAS:")
            print("-" * 40)
            
            collector.ui_manager.show_detailed_statistics(
                test_signs,
                collector.data_manager, 
                collector.sign_config
            )
            
            print("\n✅ Estadísticas detalladas mostradas")
            
            return True
            
        except RuntimeError as e:
            if 'MediaPipe' in str(e):
                print("⚠️ MediaPipe no disponible, pero estructura correcta")
                print("💡 El menú mejorado debería funcionar con los managers")
                
                # Probar solo la parte que no requiere MediaPipe
                print("\n📋 TESTING SIN MEDIAPIPE:")
                try:
                    # Crear managers básicos para testing
                    from src.data_collection.data_manager import DataManager
                    from src.data_collection.sign_config import SignConfig
                    from src.data_collection.ui_manager import UIManager
                    
                    dm = DataManager()
                    sc = SignConfig()
                    ui = UIManager()
                    
                    test_signs = ["A", "B", "C", "HOLA", "GRACIAS"]
                    
                    print("✅ Managers individuales creados")
                    
                    ui.show_menu(test_signs, dm, sc)
                    print("✅ Menú con progreso funcionando")
                    
                    ui.show_detailed_statistics(test_signs, dm, sc)
                    print("✅ Estadísticas detalladas funcionando")
                    
                    return True
                except Exception as e2:
                    print(f"❌ Error en test sin MediaPipe: {e2}")
                    return False
            else:
                raise e
                
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 VERIFICACIÓN DEL MENÚ MEJORADO")
    print("="*60)
    
    success = test_improved_menu()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ¡PRUEBA EXITOSA!")
        print("✅ Menú mejorado con indicadores de progreso funciona")
        print("✅ Estadísticas detalladas implementadas")
        print("💡 Listo para usar en el módulo de recolección")
    else:
        print("❌ Prueba falló")
        print("💡 Revisa los errores anteriores")
