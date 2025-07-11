"""
Main Data Collector - Modular Version
Clase principal que coordina todos los módulos para la recolección de datos
"""
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

# Importaciones relativas corregidas
try:
    # Intenta importación relativa primero
    from .mediapipe_manager import MediaPipeManager
    from .feature_extractor import FeatureExtractor
    from .motion_analyzer import MotionAnalyzer
    from .ui_manager import UIManager
    from .data_manager import DataManager
    from .sign_config import SignConfig
    from .data_augmentation import AugmentationIntegrator
except ImportError:
    # Si falla, intenta importación absoluta
    from src.data_collection.mediapipe_manager import MediaPipeManager
    from src.data_collection.feature_extractor import FeatureExtractor
    from src.data_collection.motion_analyzer import MotionAnalyzer
    from src.data_collection.ui_manager import UIManager
    from src.data_collection.data_manager import DataManager
    from src.data_collection.sign_config import SignConfig
    from src.data_collection.data_augmentation import AugmentationIntegrator


class LSPDataCollector:
    """
    Recolector de datos modular para Lenguaje de Señas Peruano (LSP)
    Versión 2.0 - Arquitectura modular y escalable
    """
    
    def __init__(self, sequence_length=60, num_sequences=50):
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        
        # Inicializar módulos
        self.mediapipe_manager = MediaPipeManager()
        self.feature_extractor = FeatureExtractor()
        self.motion_analyzer = MotionAnalyzer()
        self.ui_manager = UIManager()
        self.data_manager = DataManager()
        self.sign_config = SignConfig()
        
        # Inicializar augmentation integrator
        self.augmentation_integrator = AugmentationIntegrator(
            self.data_manager, self.sign_config
        )
        
        # Configurar MediaPipe
        if not self.mediapipe_manager.setup_mediapipe_tasks():
            raise RuntimeError("Error inicializando MediaPipe")
        
        self.signs_to_collect = self.sign_config.get_all_signs()
        
        print("🚀 Recolector de Datos LSP Modular Inicializado")
        print("📝 Características:")
        print("   • Arquitectura modular y escalable")
        print("   • Normalización automática derecha/izquierda")
        print("   • Detección de señas estáticas vs dinámicas")
        print("   • Soporte para 1 o 2 manos")
        print("   • Análisis de calidad en tiempo real")
        print("   • 20 métricas de movimiento optimizadas para GRU")
        print("   • Metadatos completos por secuencia")
        print("   • 🧠 Optimizado para GRU Bidireccional")
        print("   • 🎯 Secuencias de 60 frames para mejor contexto temporal")
        print("   • 📊 Features normalizadas para keras.GRU")
        print("   • 🏗️ Código modular para fácil mantenimiento")

    def collect_single_sequence(self, sign, sequence_id, collection_mode="NORMAL"):
        """Recolecta una sola secuencia para una seña específica"""
        sign_config = self.sign_config.get_sign_config(sign)
        sign_type = sign_config['sign_type']
        
        # Mostrar información de la seña
        self.ui_manager.show_collection_start(sign, sign_type, sequence_id, 
                                            self.sign_config.get_recommended_sequence_count(sign_type))
        
        print(f"\n📋 Instrucciones: {sign_config['instructions']}")
        print("💡 Consejos:")
        for tip in self.sign_config.get_learning_tips(sign):
            print(f"   • {tip}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara.")
            return False

        sequence_buffer = deque(maxlen=self.sequence_length)
        collecting = False
        frame_count = 0
        hands_info_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time.time() * 1000)
            
            # Procesar con MediaPipe
            self.mediapipe_manager.hand_landmarker.detect_async(mp_image, timestamp)
            self.mediapipe_manager.pose_landmarker.detect_async(mp_image, timestamp)
            
            # Obtener resultados
            hand_results, pose_results = self.mediapipe_manager.get_current_results()
            
            # Extraer características
            combined_data, hands_info = self.feature_extractor.extract_advanced_landmarks(
                hand_results, pose_results)
            
            # Validar ejecución de la seña
            execution_issues = self.sign_config.validate_sign_execution(hands_info, sign_config)
            
            if collecting:
                sequence_buffer.append(combined_data)
                hands_info_history.append(hands_info)
                frame_count += 1
                
                # Mostrar progreso
                self.ui_manager.draw_progress_bar(frame, frame_count, self.sequence_length)

                if frame_count >= self.sequence_length:
                    collecting = False
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Procesar y evaluar secuencia
                    return self._process_collected_sequence(
                        sequence_buffer, hands_info_history, sign, sign_type, 
                        sequence_id, collection_mode, execution_issues)

            # Dibujar interfaz
            self.ui_manager.draw_landmarks_on_frame(frame, hand_results)
            self.ui_manager.display_hud(
                frame, collecting, hands_info, self.sequence_length,
                self.feature_extractor.gru_optimized_features,
                self.feature_extractor.temporal_smoothing,
                self.feature_extractor.feature_normalization
            )
            
            # Mostrar problemas de ejecución si los hay
            if execution_issues:
                for i, issue in enumerate(execution_issues[:3]):  # Máximo 3 mensajes
                    cv2.putText(frame, f"⚠️ {issue}", (10, frame.shape[0] - 60 + i*20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            cv2.imshow('Recolector de Datos LSP - Modular', frame)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                collecting = not collecting
                if collecting:
                    frame_count = 0
                    sequence_buffer.clear()
                    hands_info_history.clear()
                    print("🎬 Iniciando recolección...")
                else:
                    print("⏸️ Recolección pausada.")
        
        cap.release()
        cv2.destroyAllWindows()
        return False

    def _process_collected_sequence(self, sequence_buffer, hands_info_history, 
                                  sign, sign_type, sequence_id, collection_mode, execution_issues):
        """Procesa una secuencia recolectada"""
        sequence_data = np.array(sequence_buffer)
        
        # Calcular métricas de movimiento
        motion_features = self.motion_analyzer.calculate_motion_features(sequence_data)
        
        # Evaluar calidad
        quality_score, quality_level, quality_issues = self.motion_analyzer.evaluate_sequence_quality(
            sequence_data, motion_features, sign_type)
        
        # Combinar problemas de ejecución y calidad
        all_issues = execution_issues + quality_issues
        
        # Mostrar resultados
        self.ui_manager.show_quality_results(quality_score, quality_level, all_issues)
        
        # Obtener confirmación del usuario
        user_choice = self.ui_manager.confirm_sequence()
        
        if user_choice == 'accept':
            # Crear metadatos
            avg_hands_info = self._average_hands_info(hands_info_history)
            metadata = self.data_manager.create_metadata(
                sign, sign_type, avg_hands_info, quality_score, 
                quality_level, motion_features, all_issues, collection_mode
            )
            
            # Guardar secuencia
            self.data_manager.save_sequence(sequence_data, sign, sequence_id, metadata)
            print(f"✅ Secuencia {sequence_id} guardada para '{sign}'.")
            return True
            
        elif user_choice == 'repeat':
            print("🔄 Repitiendo recolección...")
            return self.collect_single_sequence(sign, sequence_id, collection_mode)
        else:
            print("❌ Secuencia descartada.")
            return False

    def _average_hands_info(self, hands_info_history):
        """Calcula información promedio de las manos durante la secuencia"""
        if not hands_info_history:
            return {'count': 0, 'handedness': [], 'confidence': []}
        
        # Contar frecuencia de cada configuración
        hand_counts = [info.get('count', 0) for info in hands_info_history]
        most_common_count = max(set(hand_counts), key=hand_counts.count)
        
        # Obtener handedness más común
        all_handedness = []
        all_confidence = []
        
        for info in hands_info_history:
            if info.get('count', 0) == most_common_count:
                all_handedness.extend(info.get('handedness', []))
                all_confidence.extend(info.get('confidence', []))
        
        return {
            'count': most_common_count,
            'handedness': list(set(all_handedness)),
            'confidence': all_confidence
        }

    def collect_sign(self, sign):
        """Recolecta todas las secuencias para una seña específica"""
        sign_config = self.sign_config.get_sign_config(sign)
        recommended_count = self.sign_config.get_recommended_sequence_count(sign_config['sign_type'])
        collected_count = self.data_manager.get_collected_sequences_count(sign)
        
        print(f"\n🎯 Recolectando seña: '{sign}'")
        print(f"📊 Ya tienes: {collected_count} secuencias")
        print(f"🎯 Recomendado: {recommended_count} secuencias")
        
        if collected_count >= recommended_count:
            continue_choice = input(f"Ya tienes suficientes secuencias. ¿Continuar? (s/n): ").strip().lower()
            if continue_choice not in ['s', 'si', 'y', 'yes', '']:
                return
        
        target_sequences = max(recommended_count - collected_count, 5)  # Mínimo 5 más
        successful_collections = 0
        
        for i in range(target_sequences):
            sequence_id = self.data_manager.get_next_sequence_id(sign)
            
            print(f"\n📝 Secuencia {i+1}/{target_sequences} (ID: {sequence_id})")
            
            if self.collect_single_sequence(sign, sequence_id):
                successful_collections += 1
            
            # Preguntar si continuar
            if i < target_sequences - 1:
                continue_choice = input("\n¿Continuar con la siguiente secuencia? (s/n/q para salir): ").strip().lower()
                if continue_choice in ['n', 'no']:
                    break
                elif continue_choice in ['q', 'quit', 'salir']:
                    return
        
        # Mostrar resumen
        final_count = self.data_manager.get_collected_sequences_count(sign)
        self.ui_manager.show_collection_summary(sign, final_count, recommended_count)

    def run(self):
        """Ejecuta el bucle principal de recolección"""
        print("\n🚀 Iniciando Recolector de Datos LSP Modular")
        
        while True:
            # Mostrar menú con información de progreso
            self.ui_manager.show_menu(self.signs_to_collect, self.data_manager, self.sign_config)
            
            # Obtener selección del usuario
            choice = self.ui_manager.get_user_choice(self.signs_to_collect)
            
            if choice is None:  # Salir
                break
            elif choice == 'STATS':
                # Mostrar estadísticas detalladas
                self.ui_manager.show_detailed_statistics(
                    self.signs_to_collect, self.data_manager, self.sign_config)
                input("\n📌 Presiona Enter para continuar...")
                continue
            elif choice == 'AUGMENT':
                # Ejecutar Data Augmentation
                self._run_data_augmentation()
                continue
            elif choice == 'ALL':
                # Recolectar todas las señas
                total_target = sum(self.sign_config.get_recommended_sequence_count(
                    self.sign_config.classify_sign_type(sign)) for sign in self.signs_to_collect)
                total_collected = 0
                
                for sign in self.signs_to_collect:
                    print(f"\n{'='*50}")
                    print(f"Procesando: {sign}")
                    print(f"{'='*50}")
                    
                    initial_count = self.data_manager.get_collected_sequences_count(sign)
                    self.collect_sign(sign)
                    final_count = self.data_manager.get_collected_sequences_count(sign)
                    total_collected += (final_count - initial_count)
                    
                    # Opción de parar en cualquier momento
                    continue_all = input("\n¿Continuar con la siguiente seña? (s/n): ").strip().lower()
                    if continue_all in ['n', 'no']:
                        break
                
                self.ui_manager.show_final_summary(total_collected, total_target)
            else:
                # Recolectar seña específica
                self.collect_sign(choice)
        
        # Mostrar estadísticas finales
        self._show_final_statistics()
        print("\n👋 ¡Gracias por usar el Recolector de Datos LSP Modular!")

    def _show_final_statistics(self):
        """Muestra estadísticas finales de la sesión"""
        stats = self.data_manager.get_collection_statistics()
        
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS FINALES DEL DATASET")
        print("="*80)
        print(f"🎯 Total de señas: {stats['total_signs']}")
        print(f"📝 Total de secuencias: {stats['total_sequences']}")
        print()
        
        print("📋 Distribución por tipo:")
        for sign_type, count in stats['signs_by_type'].items():
            print(f"   • {sign_type}: {count} secuencias")
        print()
        
        print("⭐ Distribución de calidad:")
        for quality, count in stats['quality_distribution'].items():
            if count > 0:
                print(f"   • {quality}: {count} secuencias")
        print()
        
        print("📂 Estado de completación por seña:")
        for sign, count in sorted(stats['completion_status'].items()):
            recommended = self.sign_config.get_recommended_sequence_count(
                self.sign_config.classify_sign_type(sign))
            status = "✅" if count >= recommended else "⚠️"
            print(f"   {status} {sign}: {count}/{recommended}")
        
        print("="*80)
        
        # Exportar resumen
        summary_file = self.data_manager.export_dataset_summary()
        print(f"📄 Resumen exportado a: {summary_file}")
    
    def _run_data_augmentation(self):
        """Ejecuta el módulo de Data Augmentation"""
        while True:
            # Mostrar menú de augmentación
            self.ui_manager.show_augmentation_menu(
                self.signs_to_collect, self.data_manager, self.sign_config
            )
            
            # Obtener opción del usuario
            aug_choice = self.ui_manager.get_augmentation_choice()
            
            if aug_choice is None:  # Volver al menú principal
                break
            elif aug_choice == '1':
                # Augmentación conservadora (50% reducción)
                print("\n🔄 Ejecutando augmentación conservadora...")
                report = self.augmentation_integrator.auto_augment_dataset(
                    target_reduction_factor=0.5
                )
                self._show_augmentation_results(report, "CONSERVADORA")
                
            elif aug_choice == '2':
                # Augmentación moderada (70% reducción)
                print("\n🔄 Ejecutando augmentación moderada...")
                report = self.augmentation_integrator.auto_augment_dataset(
                    target_reduction_factor=0.7
                )
                self._show_augmentation_results(report, "MODERADA")
                
            elif aug_choice == '3':
                # Augmentación específica por seña
                self._run_specific_augmentation()
                
            elif aug_choice == '4':
                # Análisis detallado
                self._show_augmentation_analysis()
            
            input("\n📌 Presiona Enter para continuar...")
    
    def _show_augmentation_results(self, report: dict, mode: str):
        """Muestra los resultados de la augmentación"""
        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS AUGMENTACIÓN {mode}")
        print("="*60)
        print(f"🎯 Secuencias originales: {report['total_original']}")
        print(f"🔄 Secuencias aumentadas: {report['total_augmented']}")
        print(f"📈 Total dataset: {report['total_original'] + report['total_augmented']}")
        
        if report['total_original'] > 0:
            improvement = (report['total_augmented'] / report['total_original']) * 100
            print(f"📊 Mejora del dataset: +{improvement:.1f}%")
        
        print(f"🎯 Señas procesadas: {report['signs_processed']}")
        
        # Mostrar tiempo ahorrado estimado
        time_saved_minutes = report['total_augmented'] * 2  # 2 min por secuencia
        hours = time_saved_minutes // 60
        minutes = time_saved_minutes % 60
        print(f"⏱️ Tiempo manual ahorrado: {hours}h {minutes}m")
        
        print("="*60)
        print("✅ Augmentación completada exitosamente")
        print("💡 Puedes continuar recolectando o entrenar con el dataset expandido")
    
    def _run_specific_augmentation(self):
        """Ejecuta augmentación específica para señas seleccionadas"""
        print("\n🎯 AUGMENTACIÓN ESPECÍFICA POR SEÑA")
        print("="*50)
        
        # Mostrar señas con datos disponibles
        available_signs = []
        for sign in self.signs_to_collect:
            count = self.data_manager.get_collected_sequences_count(sign)
            if count > 0:
                available_signs.append((sign, count))
        
        if not available_signs:
            print("⚠️ No hay señas con datos para aumentar")
            print("💡 Recolecta algunas secuencias base primero")
            return
        
        print("📋 Señas disponibles para augmentación:")
        for i, (sign, count) in enumerate(available_signs, 1):
            sign_type = self.sign_config.classify_sign_type(sign)
            target = self.sign_config.get_recommended_sequence_count(sign_type)
            print(f"   {i}. {sign} - {count} secuencias (target: {target})")
        
        try:
            choice = int(input("\n🎯 Selecciona seña a aumentar (número): ")) - 1
            if 0 <= choice < len(available_signs):
                selected_sign, current_count = available_signs[choice]
                
                # Pedir cantidad de augmentaciones
                max_aug = current_count * 3  # Máximo 3 por original
                num_aug = int(input(f"🔄 Número de augmentaciones (1-{max_aug}): "))
                num_aug = min(max_aug, max(1, num_aug))
                
                print(f"\n🔄 Aumentando {selected_sign} con {num_aug} variaciones...")
                
                # Ejecutar augmentación específica
                sign_type = self.sign_config.classify_sign_type(selected_sign)
                augmented = self.augmentation_integrator._augment_sign_sequences(
                    selected_sign, sign_type, num_aug
                )
                
                print(f"✅ {selected_sign}: +{augmented} secuencias aumentadas")
                
            else:
                print("❌ Selección no válida")
        except ValueError:
            print("❌ Entrada no válida")
    
    def _show_augmentation_analysis(self):
        """Muestra análisis detallado de potencial de augmentación"""
        print("\n🔍 ANÁLISIS DETALLADO DE AUGMENTACIÓN")
        print("="*60)
        
        total_potential = 0
        total_time_saved = 0
        
        print("📊 Potencial por seña:")
        print("   Seña              | Actual | Target | Potencial | Tiempo Ahorrado")
        print("   " + "-"*70)
        
        for sign in self.signs_to_collect:
            current = self.data_manager.get_collected_sequences_count(sign)
            sign_type = self.sign_config.classify_sign_type(sign)
            target = self.sign_config.get_recommended_sequence_count(sign_type)
            
            if current > 0:
                potential = min(current * 3, target - current)  # Max 3x original
                potential = max(0, potential)
                time_saved = potential * 2  # 2 min por secuencia
                
                total_potential += potential
                total_time_saved += time_saved
                
                print(f"   {sign:<17} | {current:>6} | {target:>6} | {potential:>9} | {time_saved:>11}m")
        
        print("   " + "-"*70)
        print(f"   TOTAL             |        |        | {total_potential:>9} | {total_time_saved:>11}m")
        
        # Convertir tiempo total
        hours = total_time_saved // 60
        minutes = total_time_saved % 60
        
        print(f"\n📈 RESUMEN ANÁLISIS:")
        print(f"   🔄 Total augmentaciones posibles: {total_potential}")
        print(f"   ⏱️ Tiempo total ahorrado: {hours}h {minutes}m")
        print(f"   📊 Mejora dataset: +{total_potential} secuencias")
        
        if total_potential > 0:
            print(f"\n💡 RECOMENDACIÓN:")
            print(f"   • Ejecuta augmentación conservadora para obtener {int(total_potential * 0.5)} secuencias")
            print(f"   • Esto ahorrará aproximadamente {int(total_time_saved * 0.5)} minutos de trabajo manual")
        else:
            print(f"\n⚠️ No hay potencial de augmentación")
            print(f"   • Recolecta más secuencias base primero")

    # ...existing code...
