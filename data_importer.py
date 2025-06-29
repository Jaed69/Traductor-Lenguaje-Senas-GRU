# data_importer.py
"""
Script para importar datos de entrenamiento compartidos por otros colaboradores
y combinarlos con el dataset local.
"""

import os
import json
import shutil
import numpy as np
from datetime import datetime
import argparse

class DataImporter:
    def __init__(self, target_data_path='data/sequences'):
        self.target_path = target_data_path
        self.imported_contributors = []
        self.stats = {
            'total_imported_sequences': 0,
            'new_signs': [],
            'contributors_imported': []
        }
    
    def validate_import_data(self, import_path):
        """Valida que los datos a importar tengan el formato correcto"""
        sequences_path = os.path.join(import_path, 'sequences')
        metadata_path = os.path.join(import_path, 'metadata.json')
        
        if not os.path.exists(sequences_path):
            print(f"❌ Error: No se encontró la carpeta 'sequences' en {import_path}")
            return False
        
        if not os.path.exists(metadata_path):
            print(f"⚠️  Advertencia: No se encontró metadata.json en {import_path}")
            return True  # Continuar sin metadatos
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ Datos válidos del contribuidor: {metadata.get('contributor_id', 'unknown')}")
            return True
        except Exception as e:
            print(f"❌ Error leyendo metadatos: {e}")
            return False
    
    def load_metadata(self, import_path):
        """Carga los metadatos del contribuidor"""
        metadata_path = os.path.join(import_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def import_contributor_data(self, import_path, contributor_id=None):
        """Importa datos de un contribuidor específico"""
        if not self.validate_import_data(import_path):
            return False
        
        metadata = self.load_metadata(import_path)
        if metadata:
            contributor_id = metadata.get('contributor_id', contributor_id)
            print(f"📊 Importando datos de: {metadata.get('contributor_name', 'Anónimo')}")
            print(f"   Secuencias: {metadata.get('total_sequences', 'unknown')}")
            print(f"   Señas: {', '.join(metadata.get('signs_included', []))}")
        
        sequences_source = os.path.join(import_path, 'sequences')
        imported_sequences = 0
        new_signs = []
        
        # Crear directorio de destino si no existe
        os.makedirs(self.target_path, exist_ok=True)
        
        # Importar cada seña
        for sign_folder in os.listdir(sequences_source):
            sign_source_path = os.path.join(sequences_source, sign_folder)
            if not os.path.isdir(sign_source_path):
                continue
            
            sign_target_path = os.path.join(self.target_path, sign_folder)
            
            # Si la seña no existe, es nueva
            if not os.path.exists(sign_target_path):
                new_signs.append(sign_folder)
                os.makedirs(sign_target_path, exist_ok=True)
            
            # Contar secuencias existentes para evitar sobreescritura
            existing_sequences = len([f for f in os.listdir(sign_target_path) 
                                    if f.endswith('.npy')]) if os.path.exists(sign_target_path) else 0
            
            # Copiar nuevas secuencias
            sequence_files = [f for f in os.listdir(sign_source_path) if f.endswith('.npy')]
            for i, seq_file in enumerate(sequence_files):
                source_file = os.path.join(sign_source_path, seq_file)
                # Renombrar para evitar conflictos
                new_filename = f"{existing_sequences + i}.npy"
                target_file = os.path.join(sign_target_path, new_filename)
                
                shutil.copy2(source_file, target_file)
                imported_sequences += 1
        
        # Actualizar estadísticas
        self.stats['total_imported_sequences'] += imported_sequences
        self.stats['new_signs'].extend(new_signs)
        if contributor_id:
            self.stats['contributors_imported'].append(contributor_id)
        
        print(f"✅ Importación completada:")
        print(f"   Secuencias importadas: {imported_sequences}")
        if new_signs:
            print(f"   Nuevas señas: {', '.join(new_signs)}")
        
        return True
    
    def import_from_directory(self, shared_data_path):
        """Importa datos de múltiples contribuidores desde un directorio"""
        if not os.path.exists(shared_data_path):
            print(f"❌ Error: Directorio no encontrado: {shared_data_path}")
            return False
        
        contributors_found = []
        for item in os.listdir(shared_data_path):
            item_path = os.path.join(shared_data_path, item)
            if os.path.isdir(item_path) and item.startswith('user_'):
                contributors_found.append((item, item_path))
        
        if not contributors_found:
            print("❌ No se encontraron contribuidores en el directorio especificado")
            return False
        
        print(f"🔍 Encontrados {len(contributors_found)} contribuidores:")
        for contributor_id, _ in contributors_found:
            print(f"   - {contributor_id}")
        
        # Confirmar importación
        response = input("\n¿Desea importar todos los contribuidores? (s/N): ").lower().strip()
        if response != 's':
            print("❌ Importación cancelada")
            return False
        
        # Importar cada contribuidor
        success_count = 0
        for contributor_id, contributor_path in contributors_found:
            print(f"\n📥 Importando {contributor_id}...")
            if self.import_contributor_data(contributor_path, contributor_id):
                success_count += 1
            else:
                print(f"❌ Error importando {contributor_id}")
        
        print(f"\n🎉 Importación completada: {success_count}/{len(contributors_found)} contribuidores")
        return success_count > 0
    
    def show_import_summary(self):
        """Muestra un resumen de la importación"""
        print("\n" + "="*50)
        print("📋 RESUMEN DE IMPORTACIÓN")
        print("="*50)
        print(f"Secuencias totales importadas: {self.stats['total_imported_sequences']}")
        print(f"Contribuidores importados: {len(self.stats['contributors_imported'])}")
        if self.stats['contributors_imported']:
            print(f"IDs: {', '.join(self.stats['contributors_imported'])}")
        
        if self.stats['new_signs']:
            print(f"Nuevas señas agregadas: {', '.join(set(self.stats['new_signs']))}")
        
        # Mostrar estadísticas del dataset actual
        self.show_dataset_stats()
    
    def show_dataset_stats(self):
        """Muestra estadísticas del dataset actual"""
        if not os.path.exists(self.target_path):
            print("❌ No se encontró dataset local")
            return
        
        total_sequences = 0
        signs_stats = {}
        
        for sign_folder in os.listdir(self.target_path):
            sign_path = os.path.join(self.target_path, sign_folder)
            if os.path.isdir(sign_path):
                seq_count = len([f for f in os.listdir(sign_path) if f.endswith('.npy')])
                signs_stats[sign_folder] = seq_count
                total_sequences += seq_count
        
        print(f"\n📊 ESTADÍSTICAS DEL DATASET ACTUAL:")
        print(f"Total de señas: {len(signs_stats)}")
        print(f"Total de secuencias: {total_sequences}")
        print(f"Promedio por seña: {total_sequences/len(signs_stats):.1f}")
        
        # Mostrar señas con menos datos
        min_sequences = min(signs_stats.values()) if signs_stats else 0
        if min_sequences < 20:
            low_data_signs = [sign for sign, count in signs_stats.items() if count < 20]
            print(f"⚠️  Señas con pocos datos (<20): {', '.join(low_data_signs)}")

def main():
    parser = argparse.ArgumentParser(description='Importar datos de entrenamiento compartidos')
    parser.add_argument('--import', dest='import_path', required=True, 
                       help='Ruta al directorio de datos a importar')
    parser.add_argument('--target', default='data/sequences', 
                       help='Directorio destino para los datos')
    parser.add_argument('--contributor-id', 
                       help='ID específico del contribuidor (si no está en metadatos)')
    
    args = parser.parse_args()
    
    importer = DataImporter(args.target)
    
    print("🚀 Iniciando importación de datos...")
    
    # Verificar si es un directorio de múltiples contribuidores o uno solo
    if os.path.isdir(args.import_path):
        # Verificar si contiene subcarpetas de usuarios
        has_user_folders = any(item.startswith('user_') and os.path.isdir(os.path.join(args.import_path, item))
                             for item in os.listdir(args.import_path))
        
        if has_user_folders:
            # Directorio con múltiples contribuidores
            success = importer.import_from_directory(args.import_path)
        else:
            # Directorio de un solo contribuidor
            success = importer.import_contributor_data(args.import_path, args.contributor_id)
    else:
        print(f"❌ Error: La ruta especificada no es un directorio: {args.import_path}")
        return
    
    if success:
        importer.show_import_summary()
        print("\n💡 Consejo: Ejecuta el entrenamiento del modelo para incorporar los nuevos datos:")
        print("   python model_trainer_sequence.py")
    else:
        print("❌ La importación falló")

if __name__ == "__main__":
    main()
