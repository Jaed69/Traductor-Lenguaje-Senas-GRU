"""
User Interface and Visualization
Maneja la interfaz de usuario, visualización y controles
"""
import cv2
import numpy as np

class UIManager:
    """Maneja la interfaz de usuario y visualización para la recolección de datos."""
    
    def __init__(self):
        self.window_name = 'Recolector de Datos LSP'

    def draw_landmarks_on_frame(self, frame, hand_results):
        if not hand_results or not hand_results.hand_landmarks: return
        h, w, _ = frame.shape
        for hand_landmarks_list in hand_results.hand_landmarks:
            for i, landmark in enumerate(hand_landmarks_list):
                x, y = int(landmark.x * w), int(landmark.y * h)
                color = (0, 255, 0) if i in [4, 8, 12, 16, 20] else (255, 255, 255)
                cv2.circle(frame, (x, y), 3, color, -1)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            for start, end in connections:
                if start < len(hand_landmarks_list) and end < len(hand_landmarks_list):
                    pt1 = (int(hand_landmarks_list[start].x * w), int(hand_landmarks_list[start].y * h))
                    pt2 = (int(hand_landmarks_list[end].x * w), int(hand_landmarks_list[end].y * h))
                    cv2.line(frame, pt1, pt2, (100, 100, 255), 2)

    def display_hud(self, frame, collecting, hands_info, sequence_length, *args):
        status_text = "GRABANDO" if collecting else "PAUSADO"
        status_color = (0, 0, 255) if collecting else (255, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        hands_text = f"Manos: {hands_info.get('count', 0)}"
        if hands_info.get('handedness'): hands_text += f" ({', '.join(hands_info['handedness'])})"
        cv2.putText(frame, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_progress_bar(self, frame, frame_count, total_frames):
        if total_frames > 0:
            progress = frame_count / total_frames
            bar_width = int(progress * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0] - 10), (bar_width, frame.shape[0]), (0, 255, 0), -1)

    def draw_hands_free_status(self, frame, state, ready_feedback):
        status_map = {
            "waiting": ("ESPERANDO POSICIÓN INICIAL", (0, 255, 255)),
            "countdown": ("¡PREPÁRATE!", (0, 165, 255)),
            "collecting": ("GRABANDO...", (0, 0, 255)),
            "paused": ("PAUSADO", (255, 255, 0))
        }
        text, color = status_map.get(state, ("DESCONOCIDO", (255, 255, 255)))
        cv2.putText(frame, f"MODO AUTOMÁTICO: {text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if state == "waiting": cv2.putText(frame, ready_feedback, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_countdown(self, frame, number):
        h, w, _ = frame.shape
        text = str(number)
        font_scale, thickness = 5, 10
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pos = ((w - text_w) // 2, (h + text_h) // 2)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def draw_execution_issues(self, frame, issues):
        for i, issue in enumerate(issues[:3]):
            cv2.putText(frame, f"ADVERTENCIA: {issue}", (10, frame.shape[0] - 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    def show_menu(self, signs_to_collect, data_manager, sign_config):
        print("\n" + "="*80)
        print("🚀 RECOLECTOR DE DATOS LSP - V2.4")
        print("="*80)
        print("\n🎮 Opciones:")
        print("   [1-n] - Recolectar seña específica (Modo Manual)")
        print("   [HF]  - Iniciar recolección 'Manos Libres'")
        print("   [A]   - Data Augmentation")
        print("   [S]   - Ver estadísticas")
        print("   [Q]   - Salir")
        print("="*80)

    def get_user_choice(self, signs_to_collect):
        while True:
            choice = input("\n🎯 Selecciona una opción: ").strip().upper()
            if choice == 'Q': return None
            if choice == 'HF': return 'HANDS_FREE'
            if choice == 'A': return 'AUGMENT'
            if choice == 'S': return 'STATS'
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(signs_to_collect):
                    return signs_to_collect[idx]
                else: print("❌ Número fuera de rango.")
            except ValueError: print("❌ Entrada inválida.")

    def select_signs_for_hands_free(self, all_signs):
        print("\n✨ SELECCIÓN PARA MODO MANOS LIBRES")
        for i, sign in enumerate(all_signs): print(f"  [{i+1}] {sign}")
        print("\nEjemplo: 1, 3, 5-8, 10 | 'ALL' para todas | 'Q' para cancelar.")
        user_input = input("\n👉 Selecciona señas: ").strip().upper()
        if user_input == 'Q': return []
        if user_input == 'ALL': return all_signs
        selected_signs = set()
        try:
            for part in user_input.replace(' ', '').split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    for i in range(start, end + 1):
                        if 1 <= i <= len(all_signs): selected_signs.add(all_signs[i-1])
                else:
                    i = int(part)
                    if 1 <= i <= len(all_signs): selected_signs.add(all_signs[i-1])
            return sorted(list(selected_signs), key=all_signs.index)
        except ValueError: return []

    def show_collection_start(self, sign, sign_type, sequence_info, total_sequences, hands_free=False):
        mode = "(Modo Manos Libres)" if hands_free else "(Modo Manual)"
        print(f"\n🎯 Recolectando '{sign}' {mode} - Secuencia {sequence_info}/{total_sequences}")
        if hands_free: print("📱 Controles: [P] - Pausar/Reanudar | [Q] - Salir")
        else: print("📱 Controles: [ESPACIO] - Iniciar/Parar | [Q] - Salir")

    def show_quality_results(self, quality_score, quality_level, issues):
        print(f"\n📊 Calidad: {quality_level} ({quality_score:.1f}%) ")
        if issues: 
            print("⚠️ Puntos a mejorar:")
            for issue in issues: print(f"   • {issue}")

    def confirm_action(self, prompt):
        while True:
            response = input(f"\n{prompt} (s/n): ").strip().lower()
            if response in ['s', 'si', 'y', 'yes', '']: return True
            if response in ['n', 'no']: return False

    def confirm_sequence(self):
        while True:
            response = input("\n¿Aceptar secuencia? (s/n/r para repetir): ").strip().lower()
            if response in ['s', 'si', 'y', 'yes', '']: return 'accept'
            if response in ['n', 'no']: return 'reject'
            if response in ['r', 'repetir']: return 'repeat'

    def show_collection_summary(self, sign, collected_count, target_count):
        print(f"\n✅ Recolección completada para '{sign}'")
        print(f"📊 Secuencias recolectadas: {collected_count}/{target_count}")
        if collected_count >= target_count: print(f"🎉 ¡Meta alcanzada para '{sign}'!")
        else: print(f"⚠️ Faltan {target_count - collected_count} secuencias")

    def show_detailed_statistics(self, signs_to_collect, data_manager, sign_config, stats):
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS DETALLADAS DEL DATASET")
        print("="*80)
        print(f"🎯 Total de señas: {stats['total_signs']} | 📝 Total de secuencias: {stats['total_sequences']}")
        print("\n⭐ Distribución de calidad:")
        for quality, count in stats['quality_distribution'].items():
            if count > 0: print(f"   • {quality}: {count} secuencias")
        print("\n📂 Estado de completación por seña:")
        for sign in sorted(signs_to_collect):
            collected = data_manager.get_collected_sequences_count(sign)
            recommended = sign_config.get_recommended_sequence_count(sign_config.classify_sign_type(sign))
            status = "✅" if collected >= recommended else "⚠️"
            print(f"   {status} {sign:<15} {collected}/{recommended}")
        print("="*80)

    def show_augmentation_menu(self, signs_to_collect, data_manager, sign_config):
        print("\n" + "="*80)
        print("🔄 DATA AUGMENTATION")
        print("="*80)
        print("   [1] - Augmentación conservadora")
        print("   [2] - Augmentación moderada")
        print("   [Q] - Volver")
        print("="*80)

    def get_augmentation_choice(self):
        while True:
            choice = input("\n🔄 Selecciona tipo de augmentación: ").strip().upper()
            if choice == 'Q': return None
            if choice in ['1', '2']: return choice

    def show_augmentation_results(self, report: dict, mode: str):
        print(f"\n📊 RESULTADOS AUGMENTACIÓN {mode}")
        print(f"   • Secuencias originales: {report['total_original']}")
        print(f"   • Secuencias aumentadas: {report['total_augmented']}")
        print(f"   • Total dataset: {report['total_original'] + report['total_augmented']}")
