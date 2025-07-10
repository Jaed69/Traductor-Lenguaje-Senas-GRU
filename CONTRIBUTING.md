# 🤝 Guía de Contribución - LSP Traductor v2.0

¡Gracias por tu interés en contribuir al proyecto **Traductor de Lenguaje de Señas Peruano**! 🙌

## 🎯 **Formas de Contribuir**

### 🐛 **Reportar Bugs**
- Usar [GitHub Issues](https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU/issues)
- Incluir información del sistema (OS, Python version)
- Describir pasos para reproducir el error
- Adjuntar logs si es posible

### ✨ **Sugerir Nuevas Características**
- Crear un Issue con etiqueta "enhancement"
- Describir el problema que resuelve
- Proponer implementación si es posible
- Considerar impacto en performance

### 📝 **Mejorar Documentación**
- Corregir typos o errores
- Agregar ejemplos de uso
- Traducir a otros idiomas
- Crear tutoriales en video

### 💻 **Contribuir Código**
- Seguir las guías de estilo
- Escribir tests cuando corresponda
- Documentar funciones nuevas
- Hacer commits descriptivos

## 🔧 **Configuración para Desarrollo**

### **1. Fork y Clone:**
```bash
# Fork en GitHub primero, luego:
git clone https://github.com/TU_USUARIO/Traductor-Lenguaje-Senas-GRU.git
cd Traductor-Lenguaje-Senas-GRU

# Agregar upstream
git remote add upstream https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU.git
```

### **2. Instalar Dependencias de Desarrollo:**
```bash
pip install -r requirements.txt
pip install black flake8 pytest  # Herramientas de desarrollo
```

### **3. Crear Rama de Feature:**
```bash
git checkout -b feature/nombre-descriptivo
# o
git checkout -b fix/descripcion-del-bug
```

## 📋 **Estándares de Código**

### **Python Style Guide:**
- Seguir [PEP 8](https://pep8.org/)
- Usar `black` para formateo automático
- Máximo 88 caracteres por línea
- Docstrings en español para funciones principales

### **Convenciones de Nombres:**
```python
# Variables y funciones: snake_case
def calcular_metricas_movimiento():
    datos_secuencia = []
    
# Clases: PascalCase
class RecolectorDatosLSP:
    pass
    
# Constantes: UPPER_CASE
FRAMES_POR_SECUENCIA = 60
```

### **Estructura de Funciones:**
```python
def funcion_ejemplo(param1: str, param2: int = 10) -> list:
    """
    Descripción breve de la función.
    
    Args:
        param1: Descripción del parámetro
        param2: Descripción con valor por defecto
        
    Returns:
        Lista con resultados procesados
        
    Raises:
        ValueError: Cuando param1 está vacío
    """
    # Implementación aquí
    pass
```

## 📊 **Áreas de Contribución Prioritarias**

### **🔥 Alta Prioridad:**
1. **Optimización de Performance**
   - Reducir latencia de procesamiento
   - Optimizar uso de memoria
   - Mejorar eficiencia de algoritmos

2. **Nuevas Métricas de Calidad**
   - Métricas específicas para diferentes tipos de señas
   - Análisis de fluidez temporal
   - Detección automática de errores comunes

3. **Soporte Multi-plataforma**
   - Optimización para diferentes sistemas operativos
   - Compatibilidad con diferentes cámaras
   - Soporte para dispositivos móviles

### **⚡ Media Prioridad:**
4. **Interfaz de Usuario**
   - GUI más intuitiva
   - Visualización en tiempo real mejorada
   - Dashboard de progreso avanzado

5. **Análisis de Datos**
   - Herramientas de visualización de dataset
   - Estadísticas avanzadas de calidad
   - Exportación en diferentes formatos

6. **Integración con ML**
   - Pipelines de entrenamiento automatizados
   - Validación de datos mejorada
   - Herramientas de augmentación

### **🌟 Baja Prioridad:**
7. **Características Avanzadas**
   - Soporte para nuevos tipos de señas
   - Integración con otros datasets
   - APIs para desarrolladores externos

## 🧪 **Testing**

### **Ejecutar Tests:**
```bash
# Tests básicos
python -m pytest tests/

# Con coverage
python -m pytest --cov=src tests/

# Tests específicos
python -m pytest tests/test_landmarks.py
```

### **Crear Nuevos Tests:**
```python
# tests/test_nueva_feature.py
import pytest
from src.data_collector import LSPDataCollector

def test_nueva_funcionalidad():
    """Test para verificar nueva funcionalidad."""
    collector = LSPDataCollector()
    resultado = collector.nueva_funcion()
    assert resultado is not None
    assert len(resultado) > 0
```

## 📝 **Proceso de Pull Request**

### **1. Antes de Enviar:**
- ✅ Tests pasan localmente
- ✅ Código formateado con `black`
- ✅ No hay warnings de `flake8`
- ✅ Documentación actualizada
- ✅ CHANGELOG.md actualizado

### **2. Crear Pull Request:**
```
Título: [TIPO] Descripción breve (max 50 chars)

Descripción:
- Qué cambia y por qué
- Cómo testear los cambios
- Screenshots si aplica
- Referencias a issues (#123)

Checklist:
- [ ] Tests agregados/actualizados
- [ ] Documentación actualizada
- [ ] No rompe compatibilidad
- [ ] Performance verificada
```

### **3. Tipos de PR:**
- `[FEAT]` - Nueva característica
- `[FIX]` - Corrección de bug
- `[DOCS]` - Documentación
- `[STYLE]` - Formateo, sin cambios de lógica
- `[REFACTOR]` - Reestructuración de código
- `[PERF]` - Mejoras de performance
- `[TEST]` - Agregar/modificar tests

## 🏷️ **Versionado**

Seguimos [Semantic Versioning](https://semver.org/):

- **MAJOR** (v3.0.0): Cambios que rompen compatibilidad
- **MINOR** (v2.1.0): Nuevas características compatibles
- **PATCH** (v2.0.1): Correcciones de bugs

## 🎖️ **Reconocimiento**

Todos los contribuidores son agregados automáticamente al:
- README.md en sección "Contributors"
- CHANGELOG.md en cada release
- GitHub Contributors graph

### **Contribuidores Destacados:**
- 🥇 **Contributor del Mes**: PR más impactante
- 🐛 **Bug Hunter**: Más bugs reportados/corregidos
- 📚 **Documentation Hero**: Mejores contribuciones a docs
- 🚀 **Performance Optimizer**: Mejoras significativas de velocidad

## 📞 **Comunicación**

### **Canales Oficiales:**
- 💬 **GitHub Discussions**: Preguntas generales
- 🐛 **GitHub Issues**: Bugs y features
- 📧 **Email**: twofigsthree@gmail.com
- 💬 **Discord**: [Enlace próximamente]

### **Etiquetas de Issues:**
- `good first issue`: Ideal para principiantes
- `help wanted`: Necesitamos ayuda
- `bug`: Error confirmado
- `enhancement`: Nueva característica
- `documentation`: Mejoras en docs
- `performance`: Optimización de velocidad

## 🎉 **¡Empezar a Contribuir!**

1. **Principiantes**: Buscar issues con `good first issue`
2. **Experimentados**: Revisar `help wanted`
3. **Especialistas**: Taclear issues complejos de performance
4. **Documentadores**: Mejorar guías y tutoriales

### **Primeros Pasos Recomendados:**
- 📝 Corregir typos en documentación
- 🐛 Reportar bugs que encuentres
- 💡 Sugerir mejoras en la experiencia de usuario
- 📊 Agregar métricas de calidad nuevas

---

## 🙏 **¡Gracias por Contribuir!**

Cada contribución, sin importar el tamaño, hace que el proyecto sea mejor para toda la comunidad de investigadores y desarrolladores trabajando en reconocimiento de lenguaje de señas.

**¡Juntos podemos hacer la tecnología más accesible! 🚀🤟**
