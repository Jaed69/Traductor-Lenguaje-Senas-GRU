"""
Training Pipeline - Pipeline Completo de Entrenamiento GRU
Sistema de entrenamiento robusto para modelos de lenguaje de señas

Autor: LSP Team
Versión: 2.0 - Julio 2025
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from tensorflow import keras

from .data_loader import HDF5DataLoader
from .model_builder import GRUModelBuilder, create_optimized_gru_model


class TrainingPipeline:
    """
    Pipeline completo de entrenamiento para modelos GRU de LSP
    """
    
    def __init__(self, 
                 data_path: str = "data",
                 models_path: str = "models",
                 logs_path: str = "logs",
                 sequence_length: int = 60):
        """
        Inicializa el pipeline de entrenamiento
        
        Args:
            data_path: Ruta a los datos
            models_path: Ruta para guardar modelos
            logs_path: Ruta para logs
            sequence_length: Longitud de secuencias
        """
        self.data_path = data_path
        self.models_path = models_path
        self.logs_path = logs_path
        self.sequence_length = sequence_length
        
        # Crear directorios si no existen
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        
        # Componentes del pipeline
        self.data_loader = HDF5DataLoader(data_path, sequence_length)
        self.model_builder = GRUModelBuilder()
        self.model = None
        self.history = None
        self.training_config = {}
        
        print("🚀 PIPELINE DE ENTRENAMIENTO GRU INICIALIZADO")
        print(f"   📁 Datos: {self.data_path}")
        print(f"   💾 Modelos: {self.models_path}")
        print(f"   📊 Logs: {self.logs_path}")
    
    def prepare_data(self, 
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    random_state: int = 42,
                    normalize: bool = True) -> Dict[str, Any]:
        """
        Prepara los datos para entrenamiento
        
        Args:
            test_size: Proporción para test
            val_size: Proporción para validación
            random_state: Semilla aleatoria
            normalize: Si normalizar los datos
            
        Returns:
            Diccionario con información de preparación
        """
        print("\n📊 PREPARANDO DATOS PARA ENTRENAMIENTO...")
        
        # Verificar disponibilidad
        if not self.data_loader.check_data_availability():
            raise ValueError("Datos no disponibles")
        
        # Obtener estadísticas
        stats = self.data_loader.get_data_statistics()
        print(f"   📈 Dataset: {stats['total_sequences']} secuencias, {len(stats['signs'])} clases")
        
        # Cargar y dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_dataset(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
        
        # Normalizar si se solicita
        if normalize:
            X_train, X_val, X_test, norm_stats = self.data_loader.normalize_data(
                X_train, X_val, X_test
            )
            self.data_loader.save_preprocessing_info(norm_stats)
        else:
            norm_stats = None
        
        # Calcular pesos de clase
        class_weights = self.data_loader.get_class_weights(y_train)
        
        # Guardar datos preparados
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.class_weights = class_weights
        
        # Información de preparación
        prep_info = {
            'data_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            },
            'num_classes': len(np.unique(y_train)),
            'class_distribution': {
                'train': dict(zip(*np.unique(y_train, return_counts=True))),
                'val': dict(zip(*np.unique(y_val, return_counts=True))),
                'test': dict(zip(*np.unique(y_test, return_counts=True)))
            },
            'normalization': normalize,
            'class_weights': class_weights,
            'dataset_stats': stats
        }
        
        print("✅ Datos preparados exitosamente")
        return prep_info
    
    def build_model(self, model_config: Optional[Dict[str, Any]] = None) -> keras.Model:
        """
        Construye el modelo GRU
        
        Args:
            model_config: Configuración del modelo
            
        Returns:
            Modelo compilado
        """
        print("\n🏗️ CONSTRUYENDO MODELO GRU...")
        
        if not hasattr(self, 'X_train'):
            raise ValueError("Debes preparar los datos primero")
        
        # Configuración por defecto
        default_config = {
            'gru_units': 128,
            'num_gru_layers': 2,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.01,
            'use_attention': True
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Obtener formas de entrada
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        num_classes = len(np.unique(self.y_train))
        
        # Crear modelo
        self.model = self.model_builder.build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            **default_config
        )
        
        self.training_config.update(default_config)
        self.training_config['input_shape'] = input_shape
        self.training_config['num_classes'] = num_classes
        
        return self.model
    
    def train_model(self,
                   epochs: int = 100,
                   batch_size: int = 32,
                   patience: int = 15,
                   save_best: bool = True,
                   plot_history: bool = True) -> Dict[str, Any]:
        """
        Entrena el modelo
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            patience: Paciencia para early stopping
            save_best: Si guardar el mejor modelo
            plot_history: Si graficar el historial
            
        Returns:
            Información del entrenamiento
        """
        print(f"\n🚂 INICIANDO ENTRENAMIENTO...")
        print(f"   📊 Épocas: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   ⏰ Paciencia: {patience}")
        
        if self.model is None:
            raise ValueError("Debes construir el modelo primero")
        
        # Timestamp para archivos únicos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"gru_lsp_model_{timestamp}"
        
        # Rutas de archivos
        model_path = os.path.join(self.models_path, f"{model_name}.h5")
        log_path = os.path.join(self.logs_path, f"{model_name}_log.csv")
        
        # Crear callbacks
        callbacks = self.model_builder.create_callbacks(
            model_save_path=model_path,
            patience=patience
        )
        
        # Crear generadores de datos
        train_dataset, val_dataset = self.model_builder.get_data_generators(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            batch_size=batch_size
        )
        
        # Configuración de entrenamiento
        self.training_config.update({
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience,
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': timestamp
        })
        
        print(f"\n🎯 Comenzando entrenamiento...")
        start_time = datetime.now()
        
        # Entrenar modelo
        try:
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            training_time = datetime.now() - start_time
            print(f"\n✅ Entrenamiento completado en: {training_time}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Entrenamiento interrumpido por el usuario")
            training_time = datetime.now() - start_time
        
        except Exception as e:
            print(f"\n❌ Error durante entrenamiento: {e}")
            raise
        
        # Guardar información del entrenamiento
        training_info = {
            'model_name': model_name,
            'training_time': str(training_time),
            'final_epoch': len(self.history.history['loss']),
            'best_val_accuracy': max(self.history.history['val_accuracy']),
            'best_train_accuracy': max(self.history.history['accuracy']),
            'min_val_loss': min(self.history.history['val_loss']),
            'config': self.training_config
        }
        
        # Guardar configuración
        config_path = os.path.join(self.logs_path, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(training_info, f, indent=2, default=str)
        
        # Graficar historial si se solicita
        if plot_history and self.history:
            self.plot_training_history(model_name)\n        \n        print(f\"💾 Modelo guardado en: {model_path}\")\n        print(f\"📊 Configuración guardada en: {config_path}\")\n        \n        return training_info\n    \n    def evaluate_model(self, \n                      model_path: Optional[str] = None,\n                      detailed: bool = True) -> Dict[str, Any]:\n        \"\"\"\n        Evalúa el modelo en el conjunto de test\n        \n        Args:\n            model_path: Ruta del modelo (si None, usa el modelo actual)\n            detailed: Si mostrar evaluación detallada\n            \n        Returns:\n            Métricas de evaluación\n        \"\"\"\n        print(\"\\n🧪 EVALUANDO MODELO...\")\n        \n        # Cargar modelo si se especifica ruta\n        if model_path:\n            eval_model = keras.models.load_model(model_path)\n            print(f\"   📂 Modelo cargado desde: {model_path}\")\n        else:\n            eval_model = self.model\n            if eval_model is None:\n                raise ValueError(\"No hay modelo para evaluar\")\n        \n        # Evaluar en conjunto de test\n        test_loss, test_accuracy, test_top_k = eval_model.evaluate(\n            self.X_test, self.y_test, verbose=0\n        )\n        \n        # Predicciones detalladas\n        y_pred_proba = eval_model.predict(self.X_test, verbose=0)\n        y_pred = np.argmax(y_pred_proba, axis=1)\n        \n        # Métricas básicas\n        evaluation = {\n            'test_loss': float(test_loss),\n            'test_accuracy': float(test_accuracy),\n            'test_top_k_accuracy': float(test_top_k)\n        }\n        \n        if detailed:\n            from sklearn.metrics import classification_report, confusion_matrix\n            \n            # Reporte de clasificación\n            class_names = self.data_loader.label_encoder.classes_\n            report = classification_report(\n                self.y_test, y_pred, \n                target_names=class_names,\n                output_dict=True\n            )\n            \n            # Matriz de confusión\n            cm = confusion_matrix(self.y_test, y_pred)\n            \n            evaluation.update({\n                'classification_report': report,\n                'confusion_matrix': cm.tolist(),\n                'class_names': class_names.tolist()\n            })\n            \n            # Mostrar resultados\n            print(f\"\\n📊 RESULTADOS DE EVALUACIÓN:\")\n            print(f\"   🎯 Accuracy: {test_accuracy:.4f}\")\n            print(f\"   📉 Loss: {test_loss:.4f}\")\n            print(f\"   🏆 Top-K Accuracy: {test_top_k:.4f}\")\n            \n            print(f\"\\n📋 Por clase:\")\n            for i, class_name in enumerate(class_names):\n                precision = report[class_name]['precision']\n                recall = report[class_name]['recall']\n                f1 = report[class_name]['f1-score']\n                print(f\"   {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\")\n        \n        return evaluation\n    \n    def plot_training_history(self, model_name: str):\n        \"\"\"\n        Grafica el historial de entrenamiento\n        \n        Args:\n            model_name: Nombre del modelo para el archivo\n        \"\"\"\n        if self.history is None:\n            print(\"❌ No hay historial de entrenamiento\")\n            return\n        \n        plt.style.use('default')\n        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))\n        fig.suptitle(f'Historial de Entrenamiento - {model_name}', fontsize=16)\n        \n        # Accuracy\n        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')\n        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', color='orange')\n        ax1.set_title('Accuracy')\n        ax1.set_xlabel('Época')\n        ax1.set_ylabel('Accuracy')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # Loss\n        ax2.plot(self.history.history['loss'], label='Train Loss', color='blue')\n        ax2.plot(self.history.history['val_loss'], label='Val Loss', color='orange')\n        ax2.set_title('Loss')\n        ax2.set_xlabel('Época')\n        ax2.set_ylabel('Loss')\n        ax2.legend()\n        ax2.grid(True, alpha=0.3)\n        \n        # Learning Rate (si está disponible)\n        if 'lr' in self.history.history:\n            ax3.plot(self.history.history['lr'], label='Learning Rate', color='green')\n            ax3.set_title('Learning Rate')\n            ax3.set_xlabel('Época')\n            ax3.set_ylabel('LR')\n            ax3.set_yscale('log')\n            ax3.legend()\n            ax3.grid(True, alpha=0.3)\n        else:\n            ax3.text(0.5, 0.5, 'Learning Rate\\nno disponible', \n                    ha='center', va='center', transform=ax3.transAxes)\n        \n        # Top-K Accuracy\n        if 'top_k_categorical_accuracy' in self.history.history:\n            ax4.plot(self.history.history['top_k_categorical_accuracy'], \n                    label='Train Top-K', color='blue')\n            ax4.plot(self.history.history['val_top_k_categorical_accuracy'], \n                    label='Val Top-K', color='orange')\n            ax4.set_title('Top-K Accuracy')\n            ax4.set_xlabel('Época')\n            ax4.set_ylabel('Top-K Accuracy')\n            ax4.legend()\n            ax4.grid(True, alpha=0.3)\n        else:\n            ax4.text(0.5, 0.5, 'Top-K Accuracy\\nno disponible', \n                    ha='center', va='center', transform=ax4.transAxes)\n        \n        plt.tight_layout()\n        \n        # Guardar gráfico\n        plot_path = os.path.join(self.logs_path, f\"{model_name}_history.png\")\n        plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n        print(f\"📊 Gráficos guardados en: {plot_path}\")\n        \n        plt.show()\n    \n    def run_complete_pipeline(self, \n                            model_config: Optional[Dict[str, Any]] = None,\n                            training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n        \"\"\"\n        Ejecuta el pipeline completo de entrenamiento\n        \n        Args:\n            model_config: Configuración del modelo\n            training_config: Configuración del entrenamiento\n            \n        Returns:\n            Resultados completos\n        \"\"\"\n        print(\"🚀 EJECUTANDO PIPELINE COMPLETO DE ENTRENAMIENTO\")\n        print(\"=\" * 60)\n        \n        # Configuraciones por defecto\n        default_training = {\n            'epochs': 100,\n            'batch_size': 32,\n            'patience': 15\n        }\n        \n        if training_config:\n            default_training.update(training_config)\n        \n        try:\n            # 1. Preparar datos\n            data_info = self.prepare_data()\n            \n            # 2. Construir modelo\n            model = self.build_model(model_config)\n            \n            # 3. Entrenar modelo\n            training_info = self.train_model(**default_training)\n            \n            # 4. Evaluar modelo\n            evaluation = self.evaluate_model()\n            \n            # Resultados completos\n            results = {\n                'data_preparation': data_info,\n                'training': training_info,\n                'evaluation': evaluation,\n                'pipeline_completed': True,\n                'timestamp': datetime.now().isoformat()\n            }\n            \n            print(\"\\n🎉 PIPELINE COMPLETADO EXITOSAMENTE\")\n            print(f\"   🎯 Accuracy final: {evaluation['test_accuracy']:.4f}\")\n            print(f\"   📊 Modelo: {training_info['model_name']}\")\n            \n            return results\n            \n        except Exception as e:\n            print(f\"\\n❌ Error en el pipeline: {e}\")\n            raise\n\n\nif __name__ == \"__main__\":\n    # Ejemplo de uso del pipeline\n    print(\"🧪 EJEMPLO DE PIPELINE DE ENTRENAMIENTO\")\n    \n    # Configuración de ejemplo\n    model_config = {\n        'gru_units': 128,\n        'num_gru_layers': 2,\n        'dropout_rate': 0.3,\n        'use_attention': True\n    }\n    \n    training_config = {\n        'epochs': 50,\n        'batch_size': 16,\n        'patience': 10\n    }\n    \n    # Crear y ejecutar pipeline\n    pipeline = TrainingPipeline()\n    \n    try:\n        results = pipeline.run_complete_pipeline(\n            model_config=model_config,\n            training_config=training_config\n        )\n        print(\"\\n✅ Pipeline ejecutado exitosamente\")\n        \n    except Exception as e:\n        print(f\"\\n❌ Error: {e}\")
