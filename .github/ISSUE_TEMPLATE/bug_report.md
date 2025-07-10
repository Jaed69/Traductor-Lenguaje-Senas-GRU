---
name: 🐛 Bug Report
about: Create a report to help us improve the LSP Data Collector
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## 🐛 **Descripción del Bug**
Una descripción clara y concisa de cuál es el problema.

## 🔄 **Pasos para Reproducir**
Pasos para reproducir el comportamiento:
1. Ir a '...'
2. Hacer click en '...'
3. Desplazar hasta '...'
4. Ver error

## ✅ **Comportamiento Esperado**
Una descripción clara y concisa de lo que esperabas que pasara.

## 📸 **Screenshots**
Si es aplicable, agrega screenshots para ayudar a explicar tu problema.

## 💻 **Información del Sistema**
**Desktop (por favor completa la siguiente información):**
 - OS: [e.g. Windows 11, Ubuntu 22.04, macOS 13]
 - Python Version: [e.g. 3.11.5]
 - Cámara: [e.g. Built-in webcam, USB camera, etc.]
 - MediaPipe Version: [e.g. 0.10.11]

**Dependencias:**
```bash
# Ejecuta: pip list | grep -E "(mediapipe|opencv|numpy|scipy)"
# Y pega el resultado aquí
```

## 📋 **Logs y Errores**
```python
# Pega aquí cualquier mensaje de error o traceback
```

## 🎯 **Contexto Adicional**
Agrega cualquier otro contexto sobre el problema aquí.

### **🤟 Información Específica de LSP**
- ¿Qué tipo de seña estabas haciendo? [e.g. GRACIAS, HOLA, etc.]
- ¿El error ocurre con todas las señas o solo algunas específicas?
- ¿Cuántas personas están en el frame? [e.g. 1, 2, múltiples]
- ¿La iluminación es buena/regular/mala?

### **⚡ Información de Performance**
- ¿El sistema se vuelve lento antes del error?
- ¿Cuánta memoria RAM está usando la aplicación?
- ¿El CPU está al 100%?

### **📊 Calidad de Datos**
- ¿Cuál es el quality_score promedio antes del error?
- ¿Las métricas de shoulder_stability están bajas?
- ¿Hay problemas con hand_presence_score?

---

**Checklist:**
- [ ] He buscado en issues existentes primero
- [ ] He incluido toda la información del sistema
- [ ] He proporcionado pasos claros para reproducir
- [ ] He agregado logs/errores si están disponibles
