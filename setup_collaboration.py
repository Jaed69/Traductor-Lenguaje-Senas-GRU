# setup_collaboration.py
"""
Script de configuración para facilitar la colaboración en el proyecto.
Configura automáticamente las herramientas necesarias para compartir datos.
"""

import os
import json
from datetime import datetime

def create_collaboration_config():
    """Crea la configuración inicial para colaboración"""
    
    config = {
        "project_info": {
            "name": "Traductor de Lenguaje de Señas",
            "version": "1.0.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "collaboration_enabled": True
        },
        "data_sharing": {
            "export_format": "v1.0",
            "quality_threshold": 5.0,
            "min_sequences_per_sign": 20,
            "supported_signs": [
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                "HOLA", "GRACIAS", "POR FAVOR", "ADIOS", "SI", "NO"
            ]
        },
        "contributors": [],
        "guidelines": {
            "lighting": "Buena iluminación natural o artificial uniforme",
            "background": "Fondo uniforme, preferiblemente de color sólido",
            "camera_distance": "1-2 metros de distancia de la cámara",
            "hand_visibility": "Ambas manos completamente visibles",
            "movement_speed": "Velocidad natural, ni muy rápido ni muy lento"
        }
    }
    
    # Crear directorio de configuración
    config_dir = "collaboration_config"
    os.makedirs(config_dir, exist_ok=True)
    
    # Guardar configuración
    config_path = os.path.join(config_dir, "collaboration_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuración de colaboración creada: {config_path}")
    return config

def create_contributor_template():
    """Crea una plantilla para nuevos contribuidores"""
    template = {
        "contributor_id": "user_xxx",
        "contributor_name": "Nombre del Contribuidor",
        "email": "opcional@email.com",
        "institution": "Universidad/Organización (opcional)",
        "join_date": datetime.now().strftime("%Y-%m-%d"),
        "specializations": [],
        "equipment": {
            "camera_model": "Cámara web estándar",
            "resolution": "1920x1080",
            "fps": 30
        },
        "contributions": {
            "total_sequences": 0,
            "signs_contributed": [],
            "quality_scores": []
        },
        "notes": "Información adicional sobre el contribuidor"
    }
    
    template_path = "collaboration_config/contributor_template.json"
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Plantilla de contribuidor creada: {template_path}")
    return template

def create_quality_guidelines():
    """Crea guías de calidad para la recolección de datos"""
    guidelines = """# Guías de Calidad para Recolección de Datos

## 🎯 Objetivo
Garantizar que los datos recolectados sean consistentes y de alta calidad para mejorar el rendimiento del modelo.

## 📋 Lista de Verificación Pre-Grabación

### ✅ Configuración del Entorno
- [ ] Iluminación uniforme y suficiente
- [ ] Fondo de color sólido (preferiblemente blanco o azul claro)
- [ ] Cámara estable y bien posicionada
- [ ] Sin objetos que interfieran en el fondo

### ✅ Preparación Personal
- [ ] Ropa de color contrastante con el fondo
- [ ] Manos limpias y sin accesorios que interfieran
- [ ] Posición cómoda frente a la cámara
- [ ] Distancia apropiada (1-2 metros)

### ✅ Durante la Grabación
- [ ] Movimientos claros y bien definidos
- [ ] Velocidad natural (ni muy rápido ni muy lento)
- [ ] Completar el gesto antes de pasar al siguiente
- [ ] Mantener ambas manos visibles cuando sea necesario

## 📊 Criterios de Calidad

### Excelente (9-10 puntos)
- Gestos claros y precisos
- Iluminación óptima
- Fondo completamente uniforme
- 40+ secuencias por seña

### Buena (7-8 puntos)
- Gestos mayormente claros
- Iluminación aceptable
- Fondo mayormente uniforme
- 30-39 secuencias por seña

### Aceptable (5-6 puntos)
- Gestos reconocibles
- Iluminación variable pero suficiente
- Algunas distracciones en el fondo
- 20-29 secuencias por seña

### Necesita Mejora (<5 puntos)
- Gestos poco claros o inconsistentes
- Iluminación deficiente
- Fondo con muchas distracciones
- <20 secuencias por seña

## 🔧 Solución de Problemas Comunes

### Detección de Manos Inconsistente
- Verificar iluminación
- Limpiar lente de la cámara
- Ajustar contraste de ropa/fondo

### Landmarks Ruidosos
- Reducir movimiento de fondo
- Mejorar estabilidad de la cámara
- Verificar que las manos estén completamente visibles

### Baja Precisión del Modelo
- Incrementar variedad en los datos
- Verificar consistencia en los gestos
- Considerar regrabar señas con baja calidad

## 📈 Métricas de Éxito
- Precisión del modelo >85%
- Balance entre señas (desviación estándar <20%)
- Puntuación de calidad promedio >7
- Contribuciones de al menos 3 usuarios diferentes
"""
    
    guidelines_path = "collaboration_config/quality_guidelines.md"
    with open(guidelines_path, 'w', encoding='utf-8') as f:
        f.write(guidelines)
    
    print(f"✅ Guías de calidad creadas: {guidelines_path}")

def create_shared_directories():
    """Crea la estructura de directorios para datos compartidos"""
    directories = [
        "shared_data",
        "shared_data/merged",
        "shared_data/backups",
        "collaboration_config",
        "dataset_analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Crear README en cada directorio
        readme_path = os.path.join(directory, "README.md")
        if not os.path.exists(readme_path):
            if "shared_data" in directory:
                content = f"# {directory.replace('_', ' ').title()}\n\nEste directorio contiene datos compartidos para colaboración.\n"
            elif "collaboration_config" in directory:
                content = "# Configuración de Colaboración\n\nArchivos de configuración para el trabajo colaborativo.\n"
            else:
                content = f"# {directory.replace('_', ' ').title()}\n\nDirectorio para {directory.replace('_', ' ')}.\n"
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    print("✅ Estructura de directorios creada")

def main():
    print("🚀 Configurando herramientas de colaboración...")
    print("="*50)
    
    # Crear configuración
    create_collaboration_config()
    create_contributor_template()
    create_quality_guidelines()
    create_shared_directories()
    
    print("\n" + "="*50)
    print("✅ ¡Configuración de colaboración completada!")
    print("\n📋 Próximos pasos:")
    print("1. Revisa 'collaboration_config/quality_guidelines.md'")
    print("2. Cada colaborador debe ejecutar:")
    print("   python data_exporter.py --contributor-id user_XXX")
    print("3. Comparte los datos usando la estructura de 'shared_data/'")
    print("4. Importa datos de otros colaboradores:")
    print("   python data_importer.py --import shared_data/user_XXX/")
    print("5. Analiza el dataset combinado:")
    print("   python dataset_stats.py --visualizations")

if __name__ == "__main__":
    main()
