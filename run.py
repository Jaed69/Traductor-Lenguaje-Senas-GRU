"""
🚀 LSP (Lenguaje de Señas Peruano) - Sistema Principal
Punto de entrada principal del sistema modular de traducción de señas

Este archivo coordina todos los módulos del sistema:
- Recolección de datos
- Entrenamiento de modelos  
- Evaluación de modelos
- Traducción en tiempo real

Autor: LSP Team
Versión: 2.0 - Julio 2025
"""

import sys
import os
from typing import Optional

# Agregar el directorio src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class LSPMainSystem:
    """Sistema principal que coordina todos los módulos de LSP"""
    
    def __init__(self):
        self.modules = {
            '1': {
                'name': 'Recolección de Datos',
                'description': 'Recolectar datos de señas usando MediaPipe',
                'module': 'data_collection',
                'icon': '📊'
            },
            '2': {
                'name': 'Entrenamiento de Modelos',
                'description': 'Entrenar modelos GRU con los datos recolectados',
                'module': 'training',
                'icon': '🧠'
            },
            '3': {
                'name': 'Evaluación de Modelos',
                'description': 'Evaluar el rendimiento de modelos entrenados',
                'module': 'evaluation',
                'icon': '📈'
            },
            '4': {
                'name': 'Traducción en Tiempo Real',
                'description': 'Usar modelos entrenados para traducir señas en vivo',
                'module': 'inference',
                'icon': '🎯'
            }
        }
    
    def show_main_menu(self):
        """Muestra el menú principal del sistema"""
        print("\n" + "="*80)
        print("🚀 SISTEMA LSP - LENGUAJE DE SEÑAS PERUANO v2.0")
        print("="*80)
        print("🎯 Sistema modular de traducción de lenguaje de señas")
        print("🧠 Basado en GRU Bidireccional con MediaPipe")
        print("📅 Julio 2025 - Arquitectura Modular")
        print("\n📋 MÓDULOS DISPONIBLES:")
        print("-"*50)
        
        for key, module in self.modules.items():
            status = self._check_module_status(module['module'])
            print(f"{module['icon']} {key}. {module['name']} {status}")
            print(f"   └─ {module['description']}")
        
        print(f"\n🔧 5. Configuración del Sistema")
        print(f"ℹ️  6. Información del Proyecto")
        print(f"❌ 0. Salir")
        print("-"*50)
    
    def _check_module_status(self, module_name: str) -> str:
        """Verifica el estado de un módulo"""
        try:
            if module_name == 'data_collection':
                from src.data_collection.main_collector import LSPDataCollector
                return "✅"
            elif module_name == 'training':
                # Verificar si existe el módulo de entrenamiento
                training_path = os.path.join('src', 'training', 'train_gru.py')
                return "✅" if os.path.exists(training_path) else "⚠️"
            elif module_name == 'evaluation':
                # Verificar si existe el módulo de evaluación
                eval_path = os.path.join('src', 'evaluation', 'evaluate_model.py')
                return "✅" if os.path.exists(eval_path) else "⚠️"
            elif module_name == 'inference':
                # Verificar si existe el módulo de inferencia
                inference_path = os.path.join('src', 'inference', 'real_time_translator.py')
                return "✅" if os.path.exists(inference_path) else "⚠️"
            else:
                return "❌"
        except ImportError:
            return "❌"
    
    def run_module(self, module_key: str):
        """Ejecuta un módulo específico"""
        if module_key not in self.modules:
            print("❌ Módulo no válido")
            return
        
        module_info = self.modules[module_key]
        module_name = module_info['module']
        
        print(f"\n🚀 Iniciando {module_info['name']}...")
        print("="*60)
        
        try:
            if module_name == 'data_collection':
                self._run_data_collection()
            elif module_name == 'training':
                self._run_training()
            elif module_name == 'evaluation':
                self._run_evaluation()
            elif module_name == 'inference':
                self._run_inference()
        except ImportError as e:
            print(f"❌ Error de importación: {e}")
            print(f"💡 El módulo {module_info['name']} no está disponible o tiene dependencias faltantes")
        except Exception as e:
            print(f"❌ Error ejecutando {module_info['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_data_collection(self):
        """Ejecuta el módulo de recolección de datos"""
        from src.data_collection.main_collector import LSPDataCollector
        collector = LSPDataCollector()
        collector.run()
    
    def _run_training(self):
        """Ejecuta el módulo de entrenamiento"""
        try:
            from src.training.train_gru import GRUTrainer
            trainer = GRUTrainer()
            trainer.run()
        except ImportError:
            print("⚠️ Módulo de entrenamiento no disponible")
            print("💡 Será implementado próximamente")
    
    def _run_evaluation(self):
        """Ejecuta el módulo de evaluación"""
        try:
            from src.evaluation.evaluate_model import ModelEvaluator
            evaluator = ModelEvaluator()
            evaluator.run()
        except ImportError:
            print("⚠️ Módulo de evaluación no disponible")
            print("💡 Será implementado próximamente")
    
    def _run_inference(self):
        """Ejecuta el módulo de inferencia/traducción"""
        try:
            from src.inference.real_time_translator import RealTimeTranslator
            translator = RealTimeTranslator()
            translator.run()
        except ImportError:
            print("⚠️ Módulo de traducción no disponible")
            print("💡 Será implementado próximamente")
    
    def show_system_config(self):
        """Muestra configuración del sistema"""
        print("\n🔧 CONFIGURACIÓN DEL SISTEMA")
        print("="*50)
        print("📁 Estructura del proyecto:")
        print("   ├── data/           # Datos .npy y .json generados")
        print("   ├── models/         # Modelos entrenados (.h5) y MediaPipe (.task)")
        print("   ├── src/            # Código fuente modular")
        print("   │   ├── data_collection/   # Recolección de datos")
        print("   │   ├── training/          # Entrenamiento GRU")
        print("   │   ├── evaluation/        # Evaluación de modelos")
        print("   │   ├── inference/         # Traducción en tiempo real")
        print("   │   └── utils/             # Utilidades y descargas")
        print("   ├── tests/          # Scripts de prueba")
        print("   └── docs/           # Documentación")
        
        print("\n📋 Estado de dependencias:")
        deps_ok = self._check_dependencies()
        
        if deps_ok:
            print("\n✅ Sistema completamente configurado y listo")
        else:
            print("\n⚠️ Sistema requiere configuración adicional")
            print("💡 Ejecuta 'pip install -r requirements.txt' si hay dependencias faltantes")
    
    def _check_dependencies(self):
        """Verifica las dependencias del sistema"""
        print("📋 Verificando dependencias...")
        
        required_packages = [
            'cv2', 'mediapipe', 'numpy', 'tensorflow', 
            'matplotlib', 'seaborn', 'sklearn', 'requests'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - No instalado")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️ Faltan dependencias: {', '.join(missing_packages)}")
            print("💡 Instálalas con: pip install -r requirements.txt")
            return False
        
        # Verificar y descargar modelos MediaPipe
        print("\n🔄 Verificando modelos MediaPipe...")
        try:
            from src.utils.mediapipe_model_downloader import setup_mediapipe_models
            
            models_available = setup_mediapipe_models(
                models_dir="models", 
                auto_download=True
            )
            
            if models_available:
                print("✅ Modelos MediaPipe listos")
            else:
                print("❌ Error configurando modelos MediaPipe")
                return False
                
        except Exception as e:
            print(f"❌ Error verificando modelos MediaPipe: {e}")
            return False
        
        return True
    
    def show_project_info(self):
        """Muestra información del proyecto"""
        print("\n📘 INFORMACIÓN DEL PROYECTO")
        print("="*50)
        print("🎯 Nombre: Sistema LSP - Lenguaje de Señas Peruano")
        print("🏗️ Arquitectura: Modular con GRU Bidireccional")
        print("📅 Versión: 2.0 - Julio 2025")
        print("🧠 IA: Red GRU con MediaPipe para detección de landmarks")
        print("📊 Datos: Secuencias de 60 frames optimizadas para contexto temporal")
        print("\n🚀 Características:")
        print("   • Detección de manos y poses con MediaPipe Tasks API")
        print("   • Normalización automática derecha/izquierda")
        print("   • Análisis de calidad en tiempo real")
        print("   • Soporte para señas estáticas y dinámicas")
        print("   • Arquitectura modular y escalable")
        print("   • Optimizado para GRU Bidireccional")
        
        print("\n📋 Módulos implementados:")
        for key, module in self.modules.items():
            status = self._check_module_status(module['module'])
            print(f"   {status} {module['name']}")
    
    def run(self):
        """Función principal del sistema"""
        while True:
            try:
                self.show_main_menu()
                choice = input("\n👆 Selecciona una opción: ").strip()
                
                if choice == '0':
                    print("\n👋 ¡Hasta luego! Gracias por usar el Sistema LSP")
                    break
                elif choice in self.modules:
                    self.run_module(choice)
                elif choice == '5':
                    self.show_system_config()
                elif choice == '6':
                    self.show_project_info()
                else:
                    print("❌ Opción no válida. Por favor selecciona un número del menú.")
                
                input("\n📌 Presiona Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrumpido por el usuario")
                print("👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}")
                import traceback
                traceback.print_exc()
                input("\n📌 Presiona Enter para continuar...")


def main():
    """Punto de entrada principal"""
    try:
        print("🚀 Iniciando Sistema LSP - Versión Modular 2.0...")
        
        # Verificación automática de modelos MediaPipe al inicio
        print("🔍 Verificando configuración del sistema...")
        try:
            from src.utils.mediapipe_model_downloader import setup_mediapipe_models
            
            print("📥 Verificando modelos MediaPipe...")
            models_ready = setup_mediapipe_models(auto_download=True)
            
            if models_ready:
                print("✅ Modelos MediaPipe verificados y listos")
            else:
                print("⚠️ Advertencia: Algunos modelos MediaPipe no están disponibles")
                print("💡 El sistema puede tener funcionalidad limitada")
                
                continue_anyway = input("¿Continuar de todos modos? (s/n): ").strip().lower()
                if continue_anyway not in ['s', 'si', 'y', 'yes', '']:
                    print("❌ Inicio cancelado por el usuario")
                    return
                    
        except ImportError:
            print("⚠️ Módulo de descarga no disponible")
        except Exception as e:
            print(f"⚠️ Error verificando modelos: {e}")
            print("💡 El sistema intentará funcionar sin verificación automática")
        
        # Iniciar sistema principal
        system = LSPMainSystem()
        system.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por el usuario")
    except ImportError as e:
        print(f"\n❌ Error de importación: {e}")
        print("💡 Asegúrate de que todas las dependencias estén instaladas:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
