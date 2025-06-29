# dataset_stats.py
"""
Script para generar estadísticas detalladas del dataset de entrenamiento,
incluyendo análisis de calidad y balance de datos.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class DatasetAnalyzer:
    def __init__(self, data_path='data/sequences'):
        self.data_path = data_path
        self.stats = {}
        self.quality_metrics = {}
        
    def analyze_dataset(self):
        """Analiza el dataset completo"""
        if not os.path.exists(self.data_path):
            print(f"❌ Error: Directorio de datos no encontrado: {self.data_path}")
            return False
        
        self.stats = {
            'total_signs': 0,
            'total_sequences': 0,
            'signs_distribution': {},
            'quality_distribution': {},
            'balance_score': 0.0,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("🔍 Analizando dataset...")
        
        # Analizar cada seña
        for sign_folder in os.listdir(self.data_path):
            sign_path = os.path.join(self.data_path, sign_folder)
            if not os.path.isdir(sign_path):
                continue
                
            sequences = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            sequence_count = len(sequences)
            
            if sequence_count > 0:
                self.stats['signs_distribution'][sign_folder] = sequence_count
                self.stats['total_sequences'] += sequence_count
                self.stats['total_signs'] += 1
                
                # Analizar calidad de las secuencias
                self._analyze_sign_quality(sign_folder, sign_path, sequences)
        
        # Calcular métricas de balance
        self._calculate_balance_metrics()
        
        return True
    
    def _analyze_sign_quality(self, sign_name, sign_path, sequences):
        """Analiza la calidad de las secuencias de una seña específica"""
        sequence_lengths = []
        landmarks_variance = []
        
        for seq_file in sequences[:10]:  # Analizar solo las primeras 10 para eficiencia
            try:
                seq_path = os.path.join(sign_path, seq_file)
                sequence = np.load(seq_path)
                
                # Longitud de la secuencia
                sequence_lengths.append(len(sequence))
                
                # Varianza de los landmarks (indica movimiento)
                non_zero_frames = sequence[np.any(sequence != 0, axis=1)]
                if len(non_zero_frames) > 0:
                    variance = np.var(non_zero_frames)
                    landmarks_variance.append(variance)
                    
            except Exception as e:
                print(f"⚠️  Error analizando {seq_file}: {e}")
                continue
        
        # Calcular métricas de calidad
        avg_length = np.mean(sequence_lengths) if sequence_lengths else 0
        avg_variance = np.mean(landmarks_variance) if landmarks_variance else 0
        
        # Puntuación de calidad (0-10)
        length_score = min(10, avg_length / 5)  # Penalizar secuencias muy cortas
        movement_score = min(10, avg_variance * 1000)  # Premiar movimiento
        quantity_score = min(10, len(sequences) / 4)  # Premiar cantidad de datos
        
        quality_score = (length_score + movement_score + quantity_score) / 3
        
        self.quality_metrics[sign_name] = {
            'quality_score': quality_score,
            'avg_length': avg_length,
            'avg_variance': avg_variance,
            'sequence_count': len(sequences)
        }
    
    def _calculate_balance_metrics(self):
        """Calcula métricas de balance del dataset"""
        if not self.stats['signs_distribution']:
            return
        
        counts = list(self.stats['signs_distribution'].values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Balance score: penalizar alta variabilidad
        cv = std_count / mean_count if mean_count > 0 else 0
        balance_score = max(0, 10 - cv * 2)  # Score de 0-10
        
        self.stats['balance_score'] = balance_score
        self.stats['mean_sequences_per_sign'] = mean_count
        self.stats['std_sequences_per_sign'] = std_count
    
    def generate_report(self, output_path=None):
        """Genera un reporte detallado del dataset"""
        if not self.stats:
            print("❌ No hay estadísticas disponibles. Ejecute analyze_dataset() primero.")
            return
        
        report = []
        report.append("="*60)
        report.append("📊 REPORTE DE ANÁLISIS DEL DATASET")
        report.append("="*60)
        report.append(f"Fecha de análisis: {self.stats['analysis_date']}")
        report.append("")
        
        # Estadísticas generales
        report.append("📋 ESTADÍSTICAS GENERALES")
        report.append("-" * 30)
        report.append(f"Total de señas: {self.stats['total_signs']}")
        report.append(f"Total de secuencias: {self.stats['total_sequences']}")
        report.append(f"Promedio por seña: {self.stats.get('mean_sequences_per_sign', 0):.1f}")
        report.append(f"Puntuación de balance: {self.stats['balance_score']:.1f}/10")
        report.append("")
        
        # Distribución por señas
        report.append("📈 DISTRIBUCIÓN POR SEÑAS")
        report.append("-" * 30)
        sorted_signs = sorted(self.stats['signs_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for sign, count in sorted_signs:
            quality = self.quality_metrics.get(sign, {}).get('quality_score', 0)
            report.append(f"{sign:12} | {count:3d} secuencias | Calidad: {quality:.1f}/10")
        report.append("")
        
        # Análisis de calidad
        report.append("🎯 ANÁLISIS DE CALIDAD")
        report.append("-" * 30)
        
        if self.quality_metrics:
            avg_quality = np.mean([m['quality_score'] for m in self.quality_metrics.values()])
            report.append(f"Calidad promedio: {avg_quality:.1f}/10")
            
            # Señas con baja calidad
            low_quality_signs = [sign for sign, metrics in self.quality_metrics.items() 
                               if metrics['quality_score'] < 5]
            if low_quality_signs:
                report.append(f"Señas con baja calidad: {', '.join(low_quality_signs)}")
            
            # Señas con pocos datos
            low_data_signs = [sign for sign, count in self.stats['signs_distribution'].items() 
                            if count < 20]
            if low_data_signs:
                report.append(f"Señas con pocos datos (<20): {', '.join(low_data_signs)}")
        
        report.append("")
        
        # Recomendaciones
        report.append("💡 RECOMENDACIONES")
        report.append("-" * 30)
        
        if self.stats['balance_score'] < 7:
            report.append("• Mejorar balance: recolectar más datos para señas con pocas secuencias")
        
        if low_data_signs:
            report.append(f"• Recolectar más datos para: {', '.join(low_data_signs[:5])}")
        
        if avg_quality < 7:
            report.append("• Mejorar calidad: verificar condiciones de grabación y detección de landmarks")
        
        report.append("• Considerar validación cruzada con datos de múltiples usuarios")
        report.append("• Implementar data augmentation para señas con pocos datos")
        
        # Mostrar reporte
        report_text = "\n".join(report)
        print(report_text)
        
        # Guardar reporte si se especifica ruta
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n📄 Reporte guardado en: {output_path}")
        
        return report_text
    
    def generate_visualizations(self, output_dir='dataset_analysis'):
        """Genera visualizaciones del dataset"""
        if not self.stats:
            print("❌ No hay estadísticas disponibles.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Gráfico de distribución de señas
        plt.figure(figsize=(12, 6))
        signs = list(self.stats['signs_distribution'].keys())
        counts = list(self.stats['signs_distribution'].values())
        
        plt.bar(signs, counts)
        plt.title('Distribución de Secuencias por Seña')
        plt.xlabel('Señas')
        plt.ylabel('Número de Secuencias')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribucion_senas.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de calidad por seña
        if self.quality_metrics:
            plt.figure(figsize=(12, 6))
            quality_signs = list(self.quality_metrics.keys())
            quality_scores = [self.quality_metrics[sign]['quality_score'] for sign in quality_signs]
            
            colors = ['red' if score < 5 else 'orange' if score < 7 else 'green' for score in quality_scores]
            
            plt.bar(quality_signs, quality_scores, color=colors)
            plt.title('Puntuación de Calidad por Seña')
            plt.xlabel('Señas')
            plt.ylabel('Puntuación de Calidad (0-10)')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=7, color='blue', linestyle='--', alpha=0.7, label='Umbral recomendado')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'calidad_senas.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizaciones guardadas en: {output_dir}")
    
    def export_stats_json(self, output_path='dataset_stats.json'):
        """Exporta las estadísticas en formato JSON"""
        export_data = {
            'general_stats': self.stats,
            'quality_metrics': self.quality_metrics,
            'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Estadísticas exportadas a: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analizar estadísticas del dataset de entrenamiento')
    parser.add_argument('--data-path', default='data/sequences', help='Ruta al directorio de datos')
    parser.add_argument('--report', help='Guardar reporte en archivo de texto')
    parser.add_argument('--visualizations', action='store_true', help='Generar gráficos de análisis')
    parser.add_argument('--export-json', help='Exportar estadísticas en formato JSON')
    parser.add_argument('--output-dir', default='dataset_analysis', help='Directorio para visualizaciones')
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(args.data_path)
    
    print("🚀 Iniciando análisis del dataset...")
    
    if analyzer.analyze_dataset():
        # Generar reporte
        analyzer.generate_report(args.report)
        
        # Generar visualizaciones si se solicita
        if args.visualizations:
            try:
                analyzer.generate_visualizations(args.output_dir)
            except ImportError:
                print("⚠️  matplotlib no está disponible. Instálalo para generar visualizaciones.")
            except Exception as e:
                print(f"❌ Error generando visualizaciones: {e}")
        
        # Exportar JSON si se solicita
        if args.export_json:
            analyzer.export_stats_json(args.export_json)
        
        print("\n✅ Análisis completado exitosamente!")
    else:
        print("❌ Error durante el análisis del dataset")

if __name__ == "__main__":
    main()
