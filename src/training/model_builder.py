"""
GRU Model Builder - Arquitectura GRU Bidireccional Optimizada
Construcción de modelos GRU para clasificación de lenguaje de señas

Autor: LSP Team
Versión: 2.0 - Julio 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, Optional, Tuple
import numpy as np


class GRUModelBuilder:
    """
    Constructor de modelos GRU bidireccionales optimizados para LSP
    """
    
    def __init__(self):
        """Inicializa el constructor de modelos"""
        print("🏗️ Inicializando Constructor de Modelos GRU")
        self.model = None
        self.model_config = {}
        
    def build_model(self, 
                   input_shape: Tuple[int, int],
                   num_classes: int,
                   gru_units: int = 128,
                   num_gru_layers: int = 2,
                   dropout_rate: float = 0.3,
                   learning_rate: float = 0.001,
                   l2_reg: float = 0.01,
                   use_attention: bool = True) -> keras.Model:
        """
        Construye modelo GRU bidireccional optimizado
        
        Args:
            input_shape: Forma de entrada (secuencia_length, features)
            num_classes: Número de clases a clasificar
            gru_units: Número de unidades GRU por capa
            num_gru_layers: Número de capas GRU
            dropout_rate: Tasa de dropout
            learning_rate: Tasa de aprendizaje
            l2_reg: Regularización L2
            use_attention: Si usar mecanismo de atención
            
        Returns:
            Modelo compilado
        """
        print(f"\n🔧 CONSTRUYENDO MODELO GRU BIDIRECCIONAL")
        print(f"   📐 Input shape: {input_shape}")
        print(f"   🎯 Clases: {num_classes}")
        print(f"   🧠 Unidades GRU: {gru_units}")
        print(f"   📚 Capas GRU: {num_gru_layers}")
        print(f"   💧 Dropout: {dropout_rate}")
        print(f"   🎯 Atención: {'✅' if use_attention else '❌'}")
        
        # Configuración del modelo
        self.model_config = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'gru_units': gru_units,
            'num_gru_layers': num_gru_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'l2_reg': l2_reg,
            'use_attention': use_attention
        }
        
        # Entrada
        inputs = layers.Input(shape=input_shape, name='sequence_input')
        
        # Normalización de entrada
        x = layers.LayerNormalization(name='input_normalization')(inputs)
        
        # Capas GRU bidireccionales
        for i in range(num_gru_layers):
            return_sequences = (i < num_gru_layers - 1) or use_attention
            
            # GRU bidireccional
            x = layers.Bidirectional(
                layers.GRU(
                    gru_units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate * 0.5,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=f'gru_layer_{i+1}'
                ),
                name=f'bidirectional_gru_{i+1}'
            )(x)
            
            # Normalización y dropout adicional
            if return_sequences:
                x = layers.LayerNormalization(name=f'layer_norm_{i+1}')(x)
                x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Mecanismo de atención (opcional)
        if use_attention:
            x = self._add_attention_layer(x, gru_units * 2)  # *2 porque es bidireccional
        
        # Capas densas de clasificación
        x = layers.Dense(
            gru_units, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_hidden'
        )(x)
        
        x = layers.Dropout(dropout_rate, name='final_dropout')(x)
        
        # Capa de salida
        outputs = layers.Dense(
            num_classes, 
            activation='softmax',
            name='classification_output'
        )(x)
        
        # Crear modelo
        model = models.Model(inputs=inputs, outputs=outputs, name='GRU_LSP_Classifier')
        
        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN DEL MODELO:")
        print(f"   📈 Parámetros totales: {model.count_params():,}")
        print(f"   🎯 Parámetros entrenables: {sum([tf.reduce_prod(var.shape) for var in model.trainable_variables]):,}")
        print(f"   ⚙️ Optimizador: Adam (lr={learning_rate})")
        
        return model
    
    def _add_attention_layer(self, x, units: int):
        """
        Añade mecanismo de atención al modelo
        
        Args:
            x: Tensor de entrada
            units: Número de unidades
            
        Returns:
            Tensor con atención aplicada
        """
        # Atención simple basada en densidad
        attention_weights = layers.Dense(1, activation='tanh', name='attention_dense')(x)
        attention_weights = layers.Softmax(axis=1, name='attention_softmax')(attention_weights)
        
        # Aplicar pesos de atención
        attended = layers.Multiply(name='attention_multiply')([x, attention_weights])
        
        # Global average pooling ponderado
        output = layers.GlobalAveragePooling1D(name='attention_pooling')(attended)
        
        return output
    
    def get_model_summary(self) -> str:
        """
        Obtiene resumen detallado del modelo
        
        Returns:
            String con el resumen del modelo
        """
        if self.model is None:
            return "No hay modelo construido"
        
        import io
        import sys
        
        # Capturar output del summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary
    
    def save_model_architecture(self, filepath: str):
        """
        Guarda la arquitectura del modelo
        
        Args:
            filepath: Ruta donde guardar
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        # Guardar arquitectura en JSON
        architecture = self.model.to_json()
        
        with open(f"{filepath}_architecture.json", 'w') as f:
            f.write(architecture)
        
        # Guardar configuración personalizada
        import json
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        print(f"✅ Arquitectura guardada en: {filepath}_architecture.json")
        print(f"✅ Configuración guardada en: {filepath}_config.json")
    
    def create_callbacks(self, 
                        model_save_path: str,
                        patience: int = 15,
                        reduce_lr_patience: int = 8,
                        min_lr: float = 1e-7) -> list:
        """
        Crea callbacks para el entrenamiento
        
        Args:
            model_save_path: Ruta para guardar el mejor modelo
            patience: Paciencia para early stopping
            reduce_lr_patience: Paciencia para reducir learning rate
            min_lr: Learning rate mínimo
            
        Returns:
            Lista de callbacks
        """
        callbacks = []
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        callbacks.append(early_stopping)
        
        # Model Checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        )
        callbacks.append(checkpoint)
        
        # Reduce Learning Rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # Progress CSV Logger
        csv_logger = keras.callbacks.CSVLogger(
            f"{model_save_path.replace('.h5', '_training_log.csv')}",
            append=True
        )
        callbacks.append(csv_logger)
        
        print(f"📋 Callbacks configurados:")
        print(f"   ⏰ Early Stopping (paciencia: {patience})")
        print(f"   💾 Model Checkpoint")
        print(f"   📉 Reduce LR (paciencia: {reduce_lr_patience})")
        print(f"   📊 CSV Logger")
        
        return callbacks
    
    def get_data_generators(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray, 
                           y_val: np.ndarray,
                           batch_size: int = 32,
                           shuffle_train: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Crea generadores de datos optimizados
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            batch_size: Tamaño del batch
            shuffle_train: Si mezclar datos de entrenamiento
            
        Returns:
            Generadores de entrenamiento y validación
        """
        print(f"\n📦 CREANDO GENERADORES DE DATOS")
        print(f"   🚂 Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   ✅ Validación: {X_val.shape[0]} muestras")
        print(f"   📦 Batch size: {batch_size}")
        
        # Dataset de entrenamiento
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        if shuffle_train:
            train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Dataset de validación
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def get_model_memory_usage(self) -> Dict[str, Any]:
        """
        Calcula el uso de memoria estimado del modelo
        
        Returns:
            Diccionario con información de memoria
        """
        if self.model is None:
            return {"error": "No hay modelo construido"}
        
        # Calcular parámetros
        total_params = self.model.count_params()
        trainable_params = sum([tf.reduce_prod(var.shape) for var in self.model.trainable_variables])
        
        # Estimar memoria (4 bytes por parámetro float32)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Estimar memoria de activaciones para batch típico
        batch_size = 32
        input_shape = self.model_config.get('input_shape', (60, 100))
        activation_memory_mb = (batch_size * input_shape[0] * input_shape[1] * 4) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': int(trainable_params),
            'model_size_mb': round(model_size_mb, 2),
            'estimated_activation_memory_mb': round(activation_memory_mb, 2),
            'estimated_total_memory_mb': round(model_size_mb + activation_memory_mb * 2, 2)
        }


def create_optimized_gru_model(input_shape: Tuple[int, int], 
                              num_classes: int,
                              config: Optional[Dict[str, Any]] = None) -> keras.Model:
    """
    Función de conveniencia para crear modelo GRU optimizado
    
    Args:
        input_shape: Forma de entrada
        num_classes: Número de clases
        config: Configuración personalizada
        
    Returns:
        Modelo compilado
    """
    builder = GRUModelBuilder()
    
    # Configuración por defecto
    default_config = {
        'gru_units': 128,
        'num_gru_layers': 2,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'l2_reg': 0.01,
        'use_attention': True
    }
    
    if config:
        default_config.update(config)
    
    model = builder.build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        **default_config
    )
    
    return model, builder


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 EJEMPLO DE CONSTRUCCIÓN DE MODELO")
    
    # Parámetros de ejemplo
    sequence_length = 60
    feature_dim = 168  # Landmarks de pose + hands
    num_classes = 5
    
    input_shape = (sequence_length, feature_dim)
    
    # Crear modelo
    model, builder = create_optimized_gru_model(
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Información del modelo
    print("\n📊 INFORMACIÓN DEL MODELO:")
    memory_info = builder.get_model_memory_usage()
    for key, value in memory_info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Modelo creado exitosamente")
