# data_exporter.py
"""
Script para exportar datos de entrenamiento en formato estándar para compartir
con otros colaboradores del proyecto.
"""

import os
import json
import shutil
import numpy as np
from datetime import datetime
import argparse

class DataExporter:
    def __init__(self, source_data_path='data/sequences'):
        self.source_path = source_data_path
        self.contributor_id = None
        self.metadata = {}
        
    def set_contributor_info(self, contributor_id, contributor_name="Anonymous"):
        """Establece la información del contribuidor"""
        self.contributor_id = contributor_id
        self.metadata = {
            "contributor_id": contributor_id,
            "contributor_name": contributor_name,
            "collection_date": datetime.now().strftime("%Y-%m-%d"),
            "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sequences": 0,
            "signs_included": [],
            "camera_specs": {
                "resolution": "unknown",
                "fps": 30
            },
            "lighting_conditions": "unknown",
            "background": "unknown",
            "quality_score": 0.0,
            "notes": ""
        }
    
    def analyze_data_quality(self):
        """Analiza la calidad de los datos antes de exportar"""
        if not os.path.exists(self.source_path):
            print(f"Error: No se encontró el directorio de datos: {self.source_path}")
            return False
        
        total_sequences = 0
        signs = []
        quality_scores = []
        
        for sign_folder in os.listdir(self.source_path):
            sign_path = os.path.join(self.source_path, sign_folder)
            if os.path.isdir(sign_path):
                signs.append(sign_folder)
                sequences_count = len([f for f in os.listdir(sign_path) if f.endswith('.npy')])
                total_sequences += sequences_count
                
                # Calcular puntuación de calidad básica
                if sequences_count >= 40:
                    quality_scores.append(10)
                elif sequences_count >= 20:
                    quality_scores.append(7)
                else:
                    quality_scores.append(4)
        
        self.metadata["total_sequences"] = total_sequences
        self.metadata["signs_included"] = sorted(signs)
        self.metadata["quality_score"] = np.mean(quality_scores) if quality_scores else 0
        
        return True
    
    def export_data(self, output_path):
        """Exporta los datos al directorio especificado"""
        if not self.contributor_id:
            print("Error: Debe establecer la información del contribuidor primero")
            return False
        
        if not self.analyze_data_quality():
            return False
        
        # Crear directorio de salida
        export_dir = os.path.join(output_path, self.contributor_id)
        os.makedirs(export_dir, exist_ok=True)
        
        # Copiar datos de secuencias
        sequences_dir = os.path.join(export_dir, 'sequences')
        if os.path.exists(sequences_dir):
            shutil.rmtree(sequences_dir)
        shutil.copytree(self.source_path, sequences_dir)
        
        # Guardar metadatos
        metadata_path = os.path.join(export_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # Crear archivo README para el contribuidor
        readme_path = os.path.join(export_dir, 'README.md')
        self._create_contributor_readme(readme_path)
        
        print(f"\n✅ Datos exportados exitosamente a: {export_dir}")
        print(f"📊 Total de secuencias: {self.metadata['total_sequences']}")
        print(f"🔤 Señas incluidas: {', '.join(self.metadata['signs_included'])}")
        print(f"⭐ Puntuación de calidad: {self.metadata['quality_score']:.1f}/10")
        
        return True
    
    def _create_contributor_readme(self, readme_path):
        """Crea un README para la contribución"""
        content = f"""# Contribución de Datos - {self.contributor_id}

## Información del Contribuidor
- **ID**: {self.metadata['contributor_id']}
- **Nombre**: {self.metadata['contributor_name']}
- **Fecha de recolección**: {self.metadata['collection_date']}
- **Fecha de exportación**: {self.metadata['export_date']}

## Estadísticas del Dataset
- **Total de secuencias**: {self.metadata['total_sequences']}
- **Señas incluidas**: {', '.join(self.metadata['signs_included'])}
- **Puntuación de calidad**: {self.metadata['quality_score']:.1f}/10

## Especificaciones Técnicas
- **Resolución de cámara**: {self.metadata['camera_specs']['resolution']}
- **FPS**: {self.metadata['camera_specs']['fps']}
- **Condiciones de iluminación**: {self.metadata['lighting_conditions']}
- **Fondo**: {self.metadata['background']}

## Notas
{self.metadata['notes'] if self.metadata['notes'] else 'Ninguna nota adicional.'}

## Uso
Para importar estos datos en tu proyecto:
```bash
python data_importer.py --import {self.contributor_id}
```
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    parser = argparse.ArgumentParser(description='Exportar datos de entrenamiento para compartir')
    parser.add_argument('--contributor-id', required=True, help='ID único del contribuidor (ej: user_001)')
    parser.add_argument('--contributor-name', default='Anonymous', help='Nombre del contribuidor')
    parser.add_argument('--output', default='shared_data', help='Directorio de salida')
    parser.add_argument('--source', default='data/sequences', help='Directorio fuente de datos')
    parser.add_argument('--notes', default='', help='Notas adicionales sobre los datos')
    
    args = parser.parse_args()
    
    exporter = DataExporter(args.source)
    exporter.set_contributor_info(args.contributor_id, args.contributor_name)
    
    # Permitir al usuario agregar información adicional
    if args.notes:
        exporter.metadata['notes'] = args.notes
    
    # Interactivo: pedir información adicional
    print("🔍 Analizando calidad de los datos...")
    if exporter.analyze_data_quality():
        print(f"\n📋 Información del dataset:")
        print(f"   Secuencias totales: {exporter.metadata['total_sequences']}")
        print(f"   Señas: {', '.join(exporter.metadata['signs_included'])}")
        print(f"   Calidad estimada: {exporter.metadata['quality_score']:.1f}/10")
        
        # Preguntar por información adicional
        resolution = input("\n📷 Resolución de tu cámara (ej: 1920x1080): ").strip()
        if resolution:
            exporter.metadata['camera_specs']['resolution'] = resolution
        
        lighting = input("💡 Condiciones de iluminación (buena/regular/mala): ").strip()
        if lighting:
            exporter.metadata['lighting_conditions'] = lighting
        
        background = input("🎨 Tipo de fondo (uniforme/variable/complejo): ").strip()
        if background:
            exporter.metadata['background'] = background
        
        additional_notes = input("📝 Notas adicionales (opcional): ").strip()
        if additional_notes:
            exporter.metadata['notes'] = additional_notes
        
        # Exportar
        if exporter.export_data(args.output):
            print(f"\n🎉 ¡Listo! Ahora puedes compartir la carpeta:")
            print(f"   {os.path.join(args.output, args.contributor_id)}")
            print("\n💡 Consejo: Comprime la carpeta antes de compartirla para facilitar la descarga.")

if __name__ == "__main__":
    main()
