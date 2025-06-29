# model_trainer_sequence.py

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout

class SequenceModelTrainer:
    def __init__(self, data_path='data/sequences', model_type='gru'):
        self.data_path = data_path
        # Verifica que el directorio de datos exista
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"El directorio de datos no existe: {self.data_path}. Por favor, ejecuta primero el script de recolección.")
            
        self.signs = np.array([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))])
        self.label_encoder = LabelEncoder()
        self.model_type = model_type
        self.sequence_length = 50
        self.num_features = 21 * 3 * 2

    def load_data(self):
        sequences, labels = [], []
        
        # Primero, ajustamos el LabelEncoder con todos los nombres de las señas
        self.label_encoder.fit(self.signs)

        for sign in self.signs:
            sign_path = os.path.join(self.data_path, sign)
            for seq_file in os.listdir(sign_path):
                res = np.load(os.path.join(sign_path, seq_file))
                sequences.append(res)
                # Transformamos la etiqueta a número
                labels.append(self.label_encoder.transform([sign])[0])
        
        X = pad_sequences(sequences, maxlen=self.sequence_length, padding='post', truncating='post', dtype='float32')
        
        # ==========================================================
        # !! LA CORRECCIÓN PRINCIPAL ESTÁ AQUÍ !!
        # Convertimos las etiquetas numéricas a formato one-hot
        # ==========================================================
        y = to_categorical(labels).astype(int)
        
        # Guardar el codificador de etiquetas para usarlo después en la predicción
        np.save('data/label_encoder.npy', self.label_encoder.classes_)
        
        return X, y

    def build_model(self, input_shape, num_classes):
        model = Sequential()
        
        # Forma moderna de definir la capa de entrada (evita el UserWarning)
        model.add(Input(shape=input_shape))
        
        model.add(GRU(64, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=False, activation='relu'))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def train(self):
        X, y = self.load_data()
        
        if X.shape[0] == 0:
            print("ERROR: No se cargaron datos. El directorio 'data/sequences' podría estar vacío.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        num_classes = len(self.signs)
        model = self.build_model(input_shape=(self.sequence_length, self.num_features), num_classes=num_classes)
        
        print("Resumen del modelo:")
        print(model.summary())
        
        print("\nIniciando entrenamiento...")
        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)
        
        model.save(f'data/sign_model_{self.model_type}.h5')
        print(f"\nModelo entrenado y guardado como 'data/sign_model_{self.model_type}.h5'")

if __name__ == '__main__':
    try:
        trainer = SequenceModelTrainer(model_type='gru')
        trainer.train()
    except FileNotFoundError as e:
        print(e)