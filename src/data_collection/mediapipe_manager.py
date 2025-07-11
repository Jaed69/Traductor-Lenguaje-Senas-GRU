"""
MediaPipe Configuration and Task Setup
Maneja la inicialización y configuración de los modelos de MediaPipe
"""
import cv2
import mediapipe as mp
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeManager:
    """Gestiona la configuración y inicialización de MediaPipe"""
    
    def __init__(self):
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.latest_hand_results = None
        self.latest_pose_results = None
        self.lock = threading.Lock()
        
    def setup_mediapipe_tasks(self):
        """Inicializa los modelos de MediaPipe usando la API de Tareas."""
        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=self._process_hand_results
            )
            
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=self._process_pose_results
            )
            
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            
            print("✅ MediaPipe inicializado correctamente")
            return True
            
        except Exception as e:
            print("\n" + "="*80)
            print("❌ ERROR: No se pudieron cargar los modelos de MediaPipe.")
            print("   Asegúrate de haber descargado los archivos 'hand_landmarker.task' y 'pose_landmarker_heavy.task'")
            print("   y haberlos colocado en una carpeta llamada 'models' junto a este script.")
            print(f"   Error original: {e}")
            print("="*80 + "\n")
            return False

    def _process_hand_results(self, result, output_image, timestamp_ms: int):
        """Callback para procesar resultados de detección de manos"""
        with self.lock:
            self.latest_hand_results = result

    def _process_pose_results(self, result, output_image, timestamp_ms: int):
        """Callback para procesar resultados de detección de pose"""
        with self.lock:
            self.latest_pose_results = result
            
    def get_current_results(self):
        """Obtiene los últimos resultados de detección"""
        with self.lock:
            return self.latest_hand_results, self.latest_pose_results
