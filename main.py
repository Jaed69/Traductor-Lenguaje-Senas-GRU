# main.py

from real_time_translator import RealTimeSequenceTranslator
import os

def main():
    """Función principal para ejecutar la aplicación."""
    MODEL_PATH = 'data/sign_model_gru.h5'
    ENCODER_PATH = 'data/label_encoder.npy'

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Error: Modelo o codificador de etiquetas no encontrado.")
        print("Asegúrate de haber ejecutado 'data_collector.py' y 'model_trainer_sequence.py' primero.")
        return

    translator = RealTimeSequenceTranslator(model_path=MODEL_PATH, signs_path=ENCODER_PATH)
    translator.run()

if __name__ == '__main__':
    main()