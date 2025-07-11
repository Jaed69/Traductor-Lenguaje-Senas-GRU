
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
    from .mediapipe_manager import MediaPipeManager
    from .feature_extractor import FeatureExtractor
    from .motion_analyzer import MotionAnalyzer
    from .ui_manager import UIManager
    from .data_manager import DataManager
    from .sign_config import SignConfig
    from .data_augmentation import AugmentationIntegrator
except ImportError:
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
    Versión 2.4 - Flujo Manos Libres Corregido
    """
    
    def __init__(self, sequence_length=60, num_sequences=50):
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.mediapipe_manager = MediaPipeManager()
        self.feature_extractor = FeatureExtractor()
        self.motion_analyzer = MotionAnalyzer()
        self.ui_manager = UIManager()
        self.data_manager = DataManager()
        self.sign_config = SignConfig()
        self.augmentation_integrator = AugmentationIntegrator(self.data_manager, self.sign_config)
        if not self.mediapipe_manager.setup_mediapipe_tasks():
            raise RuntimeError("Error inicializando MediaPipe")
        self.signs_to_collect = self.sign_config.get_all_signs()
        print("🚀 Recolector de Datos LSP Modular Inicializado")
        print("📝 Características:")
        print("   • ✨ MODO MANOS LIBRES TOTALMENTE AUTOMÁTICO")
        print("   • Auto-guardado y repetición por calidad")

    def _capture_loop(self, sign, collection_mode="NORMAL", hands_free=False):
        sign_config = self.sign_config.get_sign_config(sign)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara.")
            return None, None, None

        sequence_buffer = deque(maxlen=self.sequence_length)
        hands_info_history = []
        state = "waiting"
        frame_count = 0
        countdown = 3
        last_countdown_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time.time() * 1000)
            
            self.mediapipe_manager.process_frame(mp_image, timestamp)
            hand_results, pose_results = self.mediapipe_manager.get_current_results()
            combined_data, hands_info = self.feature_extractor.extract_advanced_landmarks(hand_results, pose_results)
            execution_issues = self.sign_config.validate_sign_execution(hands_info, sign_config)

            if hands_free:
                if state == "waiting":
                    is_ready, ready_feedback = self.motion_analyzer.is_user_ready(hand_results, pose_results)
                    if is_ready:
                        state = "countdown"
                        last_countdown_time = time.time()
                        countdown = 3
                    self.ui_manager.draw_hands_free_status(frame, state, ready_feedback)
                elif state == "countdown":
                    if time.time() - last_countdown_time >= 1:
                        countdown -= 1
                        last_countdown_time = time.time()
                    self.ui_manager.draw_countdown(frame, countdown)
                    if countdown <= 0:
                        state = "collecting"
                        frame_count = 0
                        sequence_buffer.clear()
                        hands_info_history.clear()

            if state == "collecting":
                # Validar tamaño del combined_data antes de agregarlo al buffer
                if combined_data is not None and hasattr(combined_data, '__len__'):
                    # Esperamos un tamaño fijo para consistency
                    expected_size = 157  # 126 (hands) + 24 (pose) + 7 (velocity: 6 base + 1 pose_velocity)
                    if len(combined_data) != expected_size:
                        print(f"⚠️ Warning: combined_data size {len(combined_data)}, expected {expected_size}")
                        # Ajustar al tamaño esperado
                        if len(combined_data) < expected_size:
                            # Rellenar con zeros
                            padded_data = np.zeros(expected_size)
                            padded_data[:len(combined_data)] = combined_data
                            combined_data = padded_data
                        else:
                            # Truncar
                            combined_data = combined_data[:expected_size]
                    
                    sequence_buffer.append(combined_data)
                    hands_info_history.append(hands_info)
                    frame_count += 1
                    self.ui_manager.draw_progress_bar(frame, frame_count, self.sequence_length)
                    if frame_count >= self.sequence_length:
                        cap.release()
                        cv2.destroyAllWindows()
                        return sequence_buffer, hands_info_history, execution_issues
                else:
                    print("⚠️ Warning: combined_data es None o no válido, saltando frame")

            self.ui_manager.draw_landmarks_on_frame(frame, hand_results)
            self.ui_manager.display_hud(frame, state=="collecting", hands_info, self.sequence_length)
            if execution_issues: self.ui_manager.draw_execution_issues(frame, execution_issues)

            cv2.imshow('Recolector de Datos LSP', frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None, "quit"
            if not hands_free and key == ord(' '): state = "collecting" if state != "collecting" else "waiting"
            if hands_free and key == ord('p'): state = 'paused' if state != 'paused' else 'waiting'

        cap.release()
        cv2.destroyAllWindows()
        return None, None, None

    def collect_single_sequence(self, sign, sequence_id, collection_mode="NORMAL"):
        sign_config = self.sign_config.get_sign_config(sign)
        sign_type = sign_config['sign_type']
        recommended_count = self.sign_config.get_recommended_sequence_count(sign_type)
        self.ui_manager.show_collection_start(sign, sign_type, sequence_id, recommended_count)
        print(f"\n📋 Instrucciones: {sign_config['instructions']}")
        for tip in self.sign_config.get_learning_tips(sign): print(f"   • {tip}")
        sequence_buffer, hands_info_history, execution_issues = self._capture_loop(sign, collection_mode, hands_free=False)
        if sequence_buffer is None: return False
        return self._process_collected_sequence(sequence_buffer, hands_info_history, sign, sign_type, sequence_id, collection_mode, execution_issues, hands_free_mode=False)

    def run_hands_free_collection(self, signs_to_collect):
        """Ejecuta el modo de recolección manos libres con el flujo correcto de secuencias."""
        print("\n✨ Iniciando modo de recolección 'Manos Libres'")
        print("💡 Colócate en una posición neutral para empezar.")
        print("Presiona 'p' para pausar/reanudar, 'q' para salir en cualquier momento.")
        
        user_quit = False
        for sign in signs_to_collect:
            if user_quit: break

            sign_config = self.sign_config.get_sign_config(sign)
            sign_type = sign_config['sign_type']
            recommended_count = self.sign_config.get_recommended_sequence_count(sign_type)
            
            print(f"\n---\n🎯 Iniciando recolección para: '{sign.upper()}' (Objetivo: {recommended_count} secuencias)")

            while self.data_manager.get_collected_sequences_count(sign) < recommended_count:
                collected_count = self.data_manager.get_collected_sequences_count(sign)
                sequence_id = self.data_manager.get_next_sequence_id(sign)
                
                self.ui_manager.show_collection_start(sign, sign_type, collected_count + 1, recommended_count, hands_free=True)

                sequence_buffer, hands_info_history, result = self._capture_loop(sign, "HANDS_FREE", hands_free=True)

                if result == "quit":
                    print("🛑 Recolección 'Manos Libres' detenida por el usuario.")
                    user_quit = True
                    break

                if sequence_buffer:
                    self._process_collected_sequence(
                        sequence_buffer, hands_info_history, sign, sign_type,
                        sequence_id, "HANDS_FREE", result, hands_free_mode=True
                    )
                else:
                    if not user_quit:
                        print("⚠️ Error en la captura, reintentando...")
                        time.sleep(1)
            
            if not user_quit:
                print(f"✅ Seña '{sign.upper()}' completada.")

        print("\n🏁 Sesión 'Manos Libres' finalizada.")

    def _process_collected_sequence(self, sequence_buffer, hands_info_history, sign, sign_type, sequence_id, collection_mode, execution_issues, hands_free_mode=False):
        try:
            # Validar consistencia del buffer antes de crear el array
            if not sequence_buffer:
                print("❌ Error: Buffer de secuencia vacío")
                return 'reject'
            
            # Verificar que todos los elementos tienen el mismo tamaño
            buffer_sizes = [len(item) if hasattr(item, '__len__') else 0 for item in sequence_buffer]
            unique_sizes = set(buffer_sizes)
            
            if len(unique_sizes) > 1:
                print(f"⚠️ Warning: Tamaños inconsistentes detectados en buffer: {unique_sizes}")
                print("Intentando normalizar...")
                
                # Encontrar el tamaño más común
                most_common_size = max(set(buffer_sizes), key=buffer_sizes.count)
                
                # Filtrar o ajustar elementos al tamaño más común
                normalized_buffer = []
                for item in sequence_buffer:
                    if hasattr(item, '__len__') and len(item) == most_common_size:
                        normalized_buffer.append(item)
                    elif hasattr(item, '__len__') and len(item) > most_common_size:
                        # Truncar si es más largo
                        normalized_buffer.append(item[:most_common_size])
                    elif hasattr(item, '__len__') and len(item) < most_common_size:
                        # Rellenar con zeros si es más corto
                        padded_item = np.zeros(most_common_size)
                        padded_item[:len(item)] = item
                        normalized_buffer.append(padded_item)
                
                sequence_buffer = normalized_buffer
                print(f"✅ Buffer normalizado a tamaño: {most_common_size}")
            
            sequence_data = np.array(sequence_buffer)
            print(f"📊 Shape de sequence_data: {sequence_data.shape}")
            
        except Exception as e:
            print(f"❌ Error al procesar buffer de secuencia: {e}")
            print(f"Buffer info: longitud={len(sequence_buffer)}")
            if sequence_buffer:
                print(f"Primer elemento shape: {np.array(sequence_buffer[0]).shape if hasattr(sequence_buffer[0], '__len__') else 'No shape'}")
                print(f"Último elemento shape: {np.array(sequence_buffer[-1]).shape if hasattr(sequence_buffer[-1], '__len__') else 'No shape'}")
            return 'reject'
        
        motion_features = self.motion_analyzer.calculate_motion_features(sequence_data)
        quality_score, quality_level, quality_issues = self.motion_analyzer.evaluate_sequence_quality(sequence_data, motion_features, sign_type)
        all_issues = (execution_issues or []) + quality_issues
        self.ui_manager.show_quality_results(quality_score, quality_level, all_issues)
        user_choice = ''
        if hands_free_mode:
            if quality_score >= 70:
                print("✅ Calidad aceptable, guardando automáticamente...")
                user_choice = 'accept'
            else:
                print("❌ Calidad insuficiente, se repetirá la grabación.")
                user_choice = 'repeat'
            time.sleep(1.5)
        else:
            user_choice = self.ui_manager.confirm_sequence()
        
        if user_choice == 'accept':
            avg_hands_info = self._average_hands_info(hands_info_history)
            metadata = self.data_manager.create_metadata(sign, sign_type, avg_hands_info, quality_score, quality_level, motion_features, all_issues, collection_mode)
            self.data_manager.save_sequence(sequence_data, sign, sequence_id, metadata)
            return 'accept'
        elif user_choice == 'repeat':
            if not hands_free_mode: return self.collect_single_sequence(sign, sequence_id, collection_mode)
            return 'repeat'
        else:
            print("❌ Secuencia descartada.")
            return 'reject'

    def _average_hands_info(self, hands_info_history):
        if not hands_info_history: return {'count': 0, 'handedness': [], 'confidence': 0}
        hand_counts = [info.get('count', 0) for info in hands_info_history]
        most_common_count = max(set(hand_counts), key=hand_counts.count)
        all_handedness, all_confidence = [], []
        for info in hands_info_history:
            if info.get('count', 0) == most_common_count:
                all_handedness.extend(info.get('handedness', []))
                all_confidence.extend(info.get('confidence', []))
        return {'count': most_common_count, 'handedness': list(set(all_handedness)), 'confidence': np.mean(all_confidence) if all_confidence else 0}

    def collect_sign(self, sign):
        sign_config = self.sign_config.get_sign_config(sign)
        recommended_count = self.sign_config.get_recommended_sequence_count(sign_config['sign_type'])
        collected_count = self.data_manager.get_collected_sequences_count(sign)
        print(f"\n🎯 Recolectando seña: '{sign}' ({collected_count}/{recommended_count})")
        if collected_count >= recommended_count and not self.ui_manager.confirm_action("Ya tienes suficientes secuencias. ¿Continuar?"):
            return
        target_sequences = max(recommended_count - collected_count, 5)
        for i in range(target_sequences):
            sequence_id = self.data_manager.get_next_sequence_id(sign)
            self.collect_single_sequence(sign, sequence_id)
            if i < target_sequences - 1 and not self.ui_manager.confirm_action("¿Continuar con la siguiente secuencia?"):
                break
        final_count = self.data_manager.get_collected_sequences_count(sign)
        self.ui_manager.show_collection_summary(sign, final_count, recommended_count)

    def run(self):
        while True:
            self.ui_manager.show_menu(self.signs_to_collect, self.data_manager, self.sign_config)
            choice = self.ui_manager.get_user_choice(self.signs_to_collect)
            if choice is None: break
            elif choice == 'STATS': self._show_final_statistics()
            elif choice == 'AUGMENT': self._run_data_augmentation()
            elif choice == 'HANDS_FREE':
                signs_for_hf = self.ui_manager.select_signs_for_hands_free(self.signs_to_collect)
                if signs_for_hf: self.run_hands_free_collection(signs_for_hf)
            else: self.collect_sign(choice)
        print("\n👋 Saliendo del módulo de recolección.")

    def _run_data_augmentation(self):
        while True:
            self.ui_manager.show_augmentation_menu(self.signs_to_collect, self.data_manager, self.sign_config)
            aug_choice = self.ui_manager.get_augmentation_choice()
            if aug_choice is None: break
            elif aug_choice == '1':
                report = self.augmentation_integrator.auto_augment_dataset(target_reduction_factor=0.5)
                self.ui_manager.show_augmentation_results(report, "CONSERVADORA")
            elif aug_choice == '2':
                report = self.augmentation_integrator.auto_augment_dataset(target_reduction_factor=0.7)
                self.ui_manager.show_augmentation_results(report, "MODERADA")
            input("\n📌 Presiona Enter para continuar...")

    def _show_final_statistics(self):
        stats = self.data_manager.get_collection_statistics()
        self.ui_manager.show_detailed_statistics(self.signs_to_collect, self.data_manager, self.sign_config, stats)
        input("\n📌 Presiona Enter para continuar...")
