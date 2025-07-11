"""
User Interface and Visualization
Maneja la interfaz de usuario, visualización y controles
"""
import cv2
import numpy as np


class UIManager:
    """Maneja la interfaz de usuario y visualización"""
    
    def __init__(self):
        self.window_name = 'Recolector de Datos LSP'
        
    def draw_landmarks_on_frame(self, frame, hand_results):
        """Dibuja landmarks de manos en el frame"""
        if not hand_results or not hand_results.hand_landmarks:
            return
        
        # Dibujar usando cv2 directamente para máxima compatibilidad y precisión
        h, w, _ = frame.shape
        
        for hand_landmarks_list in hand_results.hand_landmarks:
            # Dibujar puntos de landmarks
            for i, landmark in enumerate(hand_landmarks_list):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Diferentes colores para diferentes tipos de puntos
                if i == 0:  # Muñeca
                    color = (0, 0, 255)  # Rojo
                    radius = 5
                elif i in [4, 8, 12, 16, 20]:  # Puntas de dedos
                    color = (0, 255, 0)  # Verde
                    radius = 4
                else:  # Otros puntos
                    color = (255, 255, 255)  # Blanco
                    radius = 3
                
                cv2.circle(frame, (x, y), radius, color, -1)
            
            # Dibujar conexiones básicas entre puntos (estructura de mano)
            connections = [
                # Pulgar
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Índice
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Medio
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Anular
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Meñique
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            for connection in connections:
                if connection[0] < len(hand_landmarks_list) and connection[1] < len(hand_landmarks_list):
                    pt1_landmark = hand_landmarks_list[connection[0]]
                    pt2_landmark = hand_landmarks_list[connection[1]]
                    
                    pt1 = (int(pt1_landmark.x * w), int(pt1_landmark.y * h))
                    pt2 = (int(pt2_landmark.x * w), int(pt2_landmark.y * h))
                    
                    cv2.line(frame, pt1, pt2, (100, 100, 255), 2)

    def display_hud(self, frame, collecting, hands_info, sequence_length=60, 
                   gru_optimized_features=True, temporal_smoothing=True, 
                   feature_normalization=True):
        """HUD optimizado para mostrar información relevante para GRU"""
        # Estado de grabación
        status_text = "GRABANDO (GRU-Optimizado)" if collecting else "PAUSADO"
        status_color = (0, 0, 255) if collecting else (255, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        
        # Información de manos detectadas
        hands_text = f"Manos: {hands_info.get('count', 0)}"
        if hands_info.get('handedness'):
            hands_text += f" ({', '.join(hands_info['handedness'])})"
        cv2.putText(frame, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Información específica para GRU
        gru_info = [
            f"Secuencia: {sequence_length} frames (GRU-opt)",
            f"Features: {'ON' if gru_optimized_features else 'OFF'}",
            f"Suavizado: {'ON' if temporal_smoothing else 'OFF'}",
            f"Normalización: {'ON' if feature_normalization else 'OFF'}"
        ]
        
        for i, info in enumerate(gru_info):
            cv2.putText(frame, info, (10, 90 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Calidad en tiempo real si está recolectando
        if collecting:
            # Indicador de estabilidad temporal
            stability_color = (0, 255, 0)  # Verde por defecto
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, stability_color, -1)
            cv2.putText(frame, "Estabilidad", (frame.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controles
        cv2.putText(frame, "ESPACIO: Iniciar/Parar | Q: Salir", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_progress_bar(self, frame, frame_count, total_frames):
        """Dibuja barra de progreso durante la recolección"""
        if total_frames > 0:
            progress_bar_width = int((frame_count / total_frames) * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0] - 10), (progress_bar_width, frame.shape[0]), (0, 255, 0), -1)

    def show_menu(self, signs_to_collect, data_manager=None, sign_config=None):
        """Muestra el menú principal de selección de señas con progreso"""
        print("\n" + "="*80)
        print("🚀 RECOLECTOR DE DATOS LSP - VERSIÓN MODULAR")
        print("="*80)
        
        # Calcular estadísticas de progreso si tenemos los managers
        total_collected = 0
        total_required = 0
        completed_signs = 0
        
        if data_manager and sign_config:
            for sign in signs_to_collect:
                collected = data_manager.get_collected_sequences_count(sign)
                sign_type = sign_config.classify_sign_type(sign)
                required = sign_config.get_recommended_sequence_count(sign_type)
                total_collected += collected
                total_required += required
                if collected >= required:
                    completed_signs += 1
        
        print("📊 PROGRESO GENERAL DEL DATASET:")
        if data_manager and sign_config:
            progress_percentage = (total_collected / total_required * 100) if total_required > 0 else 0
            print(f"   📈 Progreso total: {total_collected}/{total_required} secuencias ({progress_percentage:.1f}%)")
            print(f"   ✅ Señas completadas: {completed_signs}/{len(signs_to_collect)}")
            print(f"   ⚠️ Secuencias faltantes: {total_required - total_collected}")
            
            # Barra de progreso visual
            bar_length = 40
            filled_length = int(bar_length * progress_percentage / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"   📊 [{bar}] {progress_percentage:.1f}%")
        else:
            print("   ⚠️ Información de progreso no disponible")
        
        print("\n📋 Señas disponibles para recolectar:")
        print()
        
        # Agrupar señas por categorías
        categories = {
            'Letras estáticas (1 mano)': [s for s in signs_to_collect if len(s) == 1 and s not in ['J', 'Z', 'Ñ', 'RR', 'LL']],
            'Letras dinámicas (1 mano)': [s for s in signs_to_collect if s in ['J', 'Z', 'Ñ', 'RR', 'LL']],
            'Palabras básicas': [s for s in signs_to_collect if s in ['AMOR', 'CASA', 'FAMILIA', 'ESCUELA']],
            'Saludos y cortesía': [s for s in signs_to_collect if s in ['HOLA', 'GRACIAS', 'POR FAVOR', 'ADIÓS', 'CÓMO ESTÁS']],
            'Frases': [s for s in signs_to_collect if s in ['BUENOS DÍAS', 'BUENAS NOCHES', 'MUCHO GUSTO', 'DE NADA']]
        }
        
        sign_index = 1
        for category, signs in categories.items():
            if signs:
                print(f"📂 {category}:")
                for sign in signs:
                    # Mostrar progreso individual si tenemos los managers
                    if data_manager and sign_config:
                        collected = data_manager.get_collected_sequences_count(sign)
                        sign_type = sign_config.classify_sign_type(sign)
                        required = sign_config.get_recommended_sequence_count(sign_type)
                        status = "✅" if collected >= required else "⚠️"
                        remaining = max(0, required - collected)
                        progress_info = f" [{collected}/{required}] "
                        if remaining > 0:
                            progress_info += f"(faltan {remaining})"
                        else:
                            progress_info += "(completa)"
                    else:
                        status = "📝"
                        progress_info = ""
                    
                    print(f"   {status} {sign_index:2d}. {sign:<15} {progress_info}")
                    sign_index += 1
                print()
        
        print("🎮 Opciones:")
        print("   [1-n] - Recolectar seña específica")
        print("   [ALL] - Recolectar todas las señas")
        print("   [A]   - Data Augmentation automático")
        print("   [S]   - Ver estadísticas detalladas")
        print("   [Q]   - Salir")
        print("="*80)

    def get_user_choice(self, signs_to_collect):
        """Obtiene la selección del usuario"""
        while True:
            choice = input("\n🎯 Selecciona una opción: ").strip().upper()
            
            if choice == 'Q':
                return None
            elif choice == 'ALL':
                return 'ALL'
            elif choice == 'S':
                return 'STATS'
            elif choice == 'A':
                return 'AUGMENT'
            else:
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(signs_to_collect):
                        return signs_to_collect[choice_num - 1]
                    else:
                        print(f"❌ Número fuera de rango. Usa 1-{len(signs_to_collect)}")
                except ValueError:
                    print("❌ Entrada inválida. Usa un número, 'ALL' o 'Q'")

    def show_collection_start(self, sign, sign_type, sequence_id, total_sequences):
        """Muestra información al iniciar recolección de una seña"""
        print(f"\n🎯 Recolectando '{sign}' - Secuencia {sequence_id}/{total_sequences}")
        print(f"📝 Tipo: {sign_type}")
        print("📱 Controles:")
        print("   [ESPACIO] - Iniciar/Pausar recolección")
        print("   [Q] - Cancelar y volver al menú")
        print("   [R] - Repetir secuencia actual")

    def show_quality_results(self, quality_score, quality_level, issues):
        """Muestra resultados de evaluación de calidad"""
        print(f"\n📊 Calidad obtenida: {quality_level} ({quality_score:.1f}%)")
        if issues:
            print("⚠️ Problemas detectados:")
            for issue in issues:
                print(f"   • {issue}")

    def confirm_sequence(self):
        """Pide confirmación para guardar la secuencia"""
        while True:
            response = input("\n¿Aceptar esta secuencia? (s/n/r para repetir): ").strip().lower()
            if response in ['s', 'si', 'y', 'yes', '']:
                return 'accept'
            elif response in ['n', 'no']:
                return 'reject'
            elif response in ['r', 'repetir', 'repeat']:
                return 'repeat'
            else:
                print("❌ Respuesta inválida. Usa 's' (sí), 'n' (no) o 'r' (repetir)")

    def show_collection_summary(self, sign, collected_count, target_count):
        """Muestra resumen de recolección para una seña"""
        print(f"\n✅ Recolección completada para '{sign}'")
        print(f"📊 Secuencias recolectadas: {collected_count}/{target_count}")
        
        if collected_count >= target_count:
            print(f"🎉 ¡Meta alcanzada para '{sign}'!")
        else:
            print(f"⚠️ Faltan {target_count - collected_count} secuencias")

    def show_final_summary(self, total_collected, total_target):
        """Muestra resumen final de la sesión"""
        print("\n" + "="*80)
        print("🎉 SESIÓN DE RECOLECCIÓN COMPLETADA")
        print("="*80)
        print(f"📊 Total recolectado: {total_collected}/{total_target} secuencias")
        completion_rate = (total_collected / total_target * 100) if total_target > 0 else 0
        print(f"📈 Tasa de completación: {completion_rate:.1f}%")
        print("="*80)

    def show_detailed_statistics(self, signs_to_collect, data_manager, sign_config):
        """Muestra estadísticas detalladas del dataset"""
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS DETALLADAS DEL DATASET LSP")
        print("="*80)
        
        # Estadísticas generales
        stats = data_manager.get_collection_statistics()
        
        print("📈 RESUMEN GENERAL:")
        print(f"   🎯 Total de señas únicas: {stats['total_signs']}")
        print(f"   📝 Total de secuencias recolectadas: {stats['total_sequences']}")
        
        # Calcular progreso total
        total_required = 0
        total_collected = 0
        completed_signs = 0
        
        for sign in signs_to_collect:
            collected = data_manager.get_collected_sequences_count(sign)
            sign_type = sign_config.classify_sign_type(sign)
            required = sign_config.get_recommended_sequence_count(sign_type)
            total_collected += collected
            total_required += required
            if collected >= required:
                completed_signs += 1
        
        progress_percentage = (total_collected / total_required * 100) if total_required > 0 else 0
        remaining = total_required - total_collected
        
        print(f"   ✅ Señas completadas: {completed_signs}/{len(signs_to_collect)} ({completed_signs/len(signs_to_collect)*100:.1f}%)")
        print(f"   📊 Progreso general: {total_collected}/{total_required} ({progress_percentage:.1f}%)")
        print(f"   ⚠️ Secuencias faltantes: {remaining}")
        
        # Barra de progreso visual
        bar_length = 50
        filled_length = int(bar_length * progress_percentage / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"   📊 [{bar}] {progress_percentage:.1f}%")
        
        print("\n📋 DISTRIBUCIÓN POR CATEGORÍAS:")
        categories = {
            'Letras estáticas': [s for s in signs_to_collect if len(s) == 1 and s not in ['J', 'Z', 'Ñ', 'RR', 'LL']],
            'Letras dinámicas': [s for s in signs_to_collect if s in ['J', 'Z', 'Ñ', 'RR', 'LL']],
            'Palabras básicas': [s for s in signs_to_collect if s in ['AMOR', 'CASA', 'FAMILIA', 'ESCUELA']],
            'Saludos': [s for s in signs_to_collect if s in ['HOLA', 'GRACIAS', 'POR FAVOR', 'ADIÓS', 'CÓMO ESTÁS']],
            'Frases': [s for s in signs_to_collect if s in ['BUENOS DÍAS', 'BUENAS NOCHES', 'MUCHO GUSTO', 'DE NADA']]
        }
        
        for category, signs in categories.items():
            if signs:
                cat_collected = sum(data_manager.get_collected_sequences_count(sign) for sign in signs)
                cat_required = sum(sign_config.get_recommended_sequence_count(
                    sign_config.classify_sign_type(sign)) for sign in signs)
                cat_progress = (cat_collected / cat_required * 100) if cat_required > 0 else 0
                print(f"   📂 {category}: {cat_collected}/{cat_required} ({cat_progress:.1f}%)")
        
        print("\n⭐ DISTRIBUCIÓN POR CALIDAD:")
        for quality, count in stats['quality_distribution'].items():
            if count > 0:
                percentage = (count / stats['total_sequences'] * 100) if stats['total_sequences'] > 0 else 0
                print(f"   • {quality}: {count} secuencias ({percentage:.1f}%)")
        
        print("\n📝 ESTADO DETALLADO POR SEÑA:")
        print("   Seña              | Recolectadas | Requeridas | Estado    | Faltantes")
        print("   " + "-"*70)
        
        for sign in sorted(signs_to_collect):
            collected = data_manager.get_collected_sequences_count(sign)
            sign_type = sign_config.classify_sign_type(sign)
            required = sign_config.get_recommended_sequence_count(sign_type)
            status = "COMPLETA" if collected >= required else "PENDIENTE"
            remaining = max(0, required - collected)
            
            print(f"   {sign:<17} | {collected:>11} | {required:>9} | {status:<9} | {remaining:>8}")
        
        print("\n" + "="*80)
        
        # Recomendaciones
        if remaining > 0:
            print("💡 RECOMENDACIONES:")
            pending_signs = [sign for sign in signs_to_collect 
                           if data_manager.get_collected_sequences_count(sign) < 
                           sign_config.get_recommended_sequence_count(sign_config.classify_sign_type(sign))]
            
            if len(pending_signs) <= 5:
                print(f"   • Enfócate en completar: {', '.join(pending_signs)}")
            else:
                print(f"   • Tienes {len(pending_signs)} señas pendientes")
                print("   • Prioriza las más fáciles: letras estáticas")
            
            estimated_time = remaining * 2  # Asumiendo 2 minutos por secuencia
            hours = estimated_time // 60
            minutes = estimated_time % 60
            print(f"   • Tiempo estimado restante: {hours}h {minutes}m")
        else:
            print("🎉 ¡FELICITACIONES! Dataset completo y listo para entrenar")
        
        print("="*80)
    
    def show_augmentation_menu(self, signs_to_collect, data_manager, sign_config):
        """Muestra el menú de Data Augmentation"""
        print("\n" + "="*80)
        print("🔄 DATA AUGMENTATION - AMPLIFICADOR DE DATASET LSP")
        print("="*80)
        print("🎯 Reduce el trabajo manual usando técnicas inteligentes de augmentación")
        print("🧠 Preserva la semántica de las señas con transformaciones conservadoras")
        print()
        
        # Calcular potencial de augmentación
        stats = data_manager.get_collection_statistics()
        total_with_data = sum(1 for sign in signs_to_collect 
                             if data_manager.get_collected_sequences_count(sign) > 0)
        
        print("📊 ANÁLISIS DE AUGMENTACIÓN:")
        print(f"   📝 Secuencias actuales: {stats['total_sequences']}")
        print(f"   🎯 Señas con datos: {total_with_data}/{len(signs_to_collect)}")
        
        # Estimar potencial
        potential_augmentations = 0
        manual_work_reduction = 0
        
        for sign in signs_to_collect:
            current = data_manager.get_collected_sequences_count(sign)
            if current > 0:
                sign_type = sign_config.classify_sign_type(sign)
                target = sign_config.get_recommended_sequence_count(sign_type)
                deficit = max(0, target - current)
                augmentable = min(deficit, current * 3)  # Max 3 augmentaciones por original
                potential_augmentations += augmentable
                manual_work_reduction += augmentable
        
        print(f"   🔄 Augmentaciones posibles: +{potential_augmentations}")
        print(f"   ⚡ Reducción trabajo manual: {manual_work_reduction} secuencias")
        
        if potential_augmentations > 0:
            time_saved = manual_work_reduction * 2  # 2 minutos por secuencia
            hours_saved = time_saved // 60
            minutes_saved = time_saved % 60
            print(f"   ⏱️ Tiempo ahorrado estimado: {hours_saved}h {minutes_saved}m")
        
        print("\n🔧 TÉCNICAS DE AUGMENTACIÓN DISPONIBLES:")
        print("   🔄 Variaciones temporales: velocidad, pausas, interpolación")
        print("   🔄 Transformaciones espaciales: rotación, escala, traslación")
        print("   🔄 Ruido controlado: gaussiano, jitter, dropout landmarks")
        print("   🔄 Variaciones de manos: intercambio izq/der")
        
        print("\n🎮 OPCIONES DE AUGMENTACIÓN:")
        print("   [1] - Augmentación conservadora (50% reducción manual)")
        print("   [2] - Augmentación moderada (70% reducción manual)")
        print("   [3] - Augmentación específica por seña")
        print("   [4] - Análisis detallado de augmentación")
        print("   [Q] - Volver al menú principal")
        print("="*80)
    
    def get_augmentation_choice(self):
        """Obtiene la opción de augmentación del usuario"""
        while True:
            choice = input("\n🔄 Selecciona tipo de augmentación: ").strip().upper()
            
            if choice == 'Q':
                return None
            elif choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ Opción no válida. Selecciona 1, 2, 3, 4 o Q")
