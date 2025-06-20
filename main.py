# main.py

from real_time_translator import RealTimeTranslator
import os

def main():
    """Función principal para ejecutar la aplicación."""
    MODEL_PATH = 'data/sign_model.pkl'
    ENCODER_PATH = 'data/label_encoder.pkl'

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Error: Modelo o codificador de etiquetas no encontrado.")
        print("Asegúrate de haber ejecutado 'data_collector.py' y 'model_trainer.py' primero.")
        return

    translator = RealTimeTranslator(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
    translator.run()

if __name__ == '__main__':
    main()