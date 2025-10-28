# 🚀 Naturgy AI Incident Classifier

Sistema de Inteligencia Artificial para clasificación automática de incidencias del sistema Delta de Naturgy.

## 📋 Descripción del Proyecto

Este proyecto desarrolla un sistema completo de IA que automatiza la clasificación de incidencias mediante:

- **Análisis automático de texto** con preprocesamiento específico para Naturgy
- **Clustering inteligente** que identifica categorías específicas de incidencias
- **Modelo predictivo** con alta precisión para clasificación automática
- **Sistema de reglas semánticas** para clasificación determinística
- **API lista para producción** e integración con ITSM Remedy

## 🎯 Características Principales

- ✅ **Clasificación automática** con 24 categorías técnicas específicas
- ✅ **Reglas semánticas avanzadas** para alta precisión
- ✅ **Sistema de confianza** con evaluación de incertidumbre
- ✅ **Estructura organizada** de salida con reportes categorizados
- ✅ **Generación de casos de prueba** para validación
- ✅ **Nomenclatura semánticamente coherente** para categorías

## 📁 Estructura del Proyecto

```
naturgy-delta-incidents-classification/
├── src/                                    # Código fuente principal
│   ├── naturgy_classifier_refactored.py   # Clasificador principal refactorizado
│   ├── semantic_analyzer_refactored.py    # Analizador semántico adicional
│   └── test_classifier.py                 # Generador de casos de prueba
├── docs/                                   # Documentación
│   ├── README.md                          # Documentación técnica detallada
│   ├── resumen_ejecutivo_proyecto_naturgy.txt
│   ├── explicacion_centroides_naturgy.txt
│   └── flujo_datos_completo.txt
├── outputs/                                # Resultados del análisis (generado)
│   ├── models/                            # Modelos entrenados
│   ├── reports/                           # Reportes de texto
│   ├── data/                              # Datos JSON de análisis
│   └── logs/                              # Archivos de registro
├── casos_prueba/                           # Casos de prueba (generado)
│   ├── models/                            # Modelo para casos de prueba
│   ├── reports/                           # Reportes de casos de prueba
│   └── data/                              # Resultados JSON de pruebas
├── infomation.xlsx                         # Dataset original de incidencias
├── README.md                              # Este archivo
└── .gitignore                             # Archivos ignorados por Git
```

## 🚀 Uso Rápido

### Instalación de Dependencias

```bash
# Dependencias básicas
pip install pandas numpy scikit-learn openpyxl

# Dependencias opcionales para NLP avanzado
pip install nltk
```

### 1. Análisis Completo Estándar

```bash
python3 src/naturgy_classifier_refactored.py infomation.xlsx
```

Este comando:
- Crea la estructura de carpetas `outputs/`
- Entrena el modelo con todos los datos
- Genera reportes completos y modelo entrenado

### 2. Análisis con Casos de Prueba para Validación

```bash
python3 src/test_classifier.py infomation.xlsx
```

Este comando:
- Separa 100 casos como prueba
- Entrena con el resto de datos
- Clasifica los casos de prueba
- Genera reporte de incertidumbre
- Crea carpeta `casos_prueba/`

### 3. Análisis Personalizado con Nombre

```bash
# Opción 1: Con fecha automática (Linux/macOS)
python3 src/naturgy_classifier_refactored.py infomation.xlsx mi_analisis_$(date +%Y_%m_%d)

# Opción 2: Con nombre personalizado (todas las plataformas)
python3 src/naturgy_classifier_refactored.py infomation.xlsx mi_analisis_custom
```

## 📊 Sistema de Clasificación

### Categorías Técnicas Identificadas (24 principales)

1. **Gestión de CUPS** - Todo ticket con códigos CUPS y acciones asociadas
2. **Montaje/Desmontaje/Equipos** - Aparatos, contadores, equipos de medida
3. **Errores de Cálculo/Facturación** - Fallos en procesos de cálculo
4. **Estados de Cálculo** - Cambios de estado (calculable, bloqueado, etc.)
5. **Lecturas y Mediciones** - Problemas con lecturas y medición
6. **Direcciones y Datos Cliente** - Cambios en información personal
7. **Cambio de Titularidad** - Transferencias de contratos
8. **Ofertas y Contratación** - Gestión de ofertas comerciales
9. **Tarifas y Productos** - Modificaciones de tarifas y servicios
10. **Gestión de Contratos** - Actualización de condiciones contractuales
11. **Bono Social y Vulnerabilidad** - Gestión de bonificaciones sociales
12. **Rechazos y Bloqueos** - Solicitudes rechazadas o bloqueadas
13. **Cobros y Pagos** - Problemas financieros y de pagos
14. **Batch/Procesos Automáticos** - Errores en procesos batch
15. **Extracciones e Informes** - Solicitudes de datos y reportes
16. **Telemedida** - Medición remota y telemedida
17. **Errores XML/Mensajería** - Problemas de comunicación
18. **Integraciones Externas** - Conexiones con sistemas externos
19. **Campañas y Marketing** - Gestión de campañas comerciales
20. **Plantillas y Documentación** - Gestión documental
21. **Consultas Funcionales** - Preguntas sobre funcionamiento
22. **Gestión de Usuarios** - Permisos y accesos del sistema
23. **Gestiones Administrativas** - Tareas internas administrativas
24. **Sincronización de Datos** - Replicación entre sistemas

### Sistema de Confianza

- **Alta confianza (>0.8)**: Clasificación automática segura
- **Media confianza (0.5-0.8)**: Requiere validación
- **Baja confianza (<0.5)**: Requiere revisión manual
- **Sin determinar (<0.76)**: Casos de incertidumbre para análisis

## 🔧 Arquitectura Técnica

### Componentes Principales

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  OutputManager  │    │ CategoryNaming   │    │ TextPreprocessor│
│  - Estructura   │    │ Engine           │    │ - Limpieza      │
│  - Organización │    │ - Nomenclatura   │    │ - Normalización │
│  - Archivos     │    │ - Semántica      │    │ - Sinonimias    │
└─────────────────┘    └──────────────────┘    └─────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ EntityExtractor │    │ IncidentClusterer│    │PredictiveClassifier│
│ - CUPS          │    │ - K-Means        │    │ - Random Forest │
│ - Códigos       │    │ - Optimización   │    │ - Features      │
│ - Fechas        │    │ - Validación     │    │ - Evaluación    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Flujo de Procesamiento

```
Datos → Preprocesamiento → Extracción Entidades → Reglas Semánticas
                                                        ↓
Clustering ← Entrenamiento Modelo ← Features Engineering
    ↓
Clasificación Automática → Evaluación → Reportes
```

## 🎛️ Configuración

El sistema incluye configuración por defecto optimizada:

```python
config = {
    'max_clusters': 50,           # Máximo número de clusters
    'min_cluster_size': 20,       # Tamaño mínimo de cluster
    'tfidf_max_features': 8000,   # Features máximas TF-IDF
    'tfidf_min_df': 3,            # Frecuencia mínima de términos
    'tfidf_max_df': 0.7,          # Frecuencia máxima de términos
    'model_type': 'random_forest', # Tipo de modelo
    'cv_folds': 5,                # Pliegues para validación cruzada
    'use_hierarchical': True,     # Clustering jerárquico
    'silhouette_threshold': 0.1   # Umbral de calidad
}
```

## 📈 Salida del Sistema

### Estructura de Archivos Generados

**Carpeta `outputs/`:**
- `models/naturgy_model_[timestamp].pkl` - Modelo entrenado
- `data/analisis_completo_naturgy.json` - Análisis completo en JSON  
- `reports/reporte_analisis_naturgy.txt` - Reporte legible detallado
- `reports/resumen_ejecutivo.txt` - Resumen para management

**Carpeta `casos_prueba/` (con test_classifier.py):**
- `data/casos_prueba_original.xlsx` - Casos de prueba originales
- `reports/casos_prueba_detallado.txt` - Reporte de clasificación
- `reports/incertidumbre.txt` - Casos con baja confianza
- `data/casos_prueba_resultados.json` - Resultados JSON

## 🤖 Uso Programático

### Clasificar Nueva Incidencia

```python
from src.naturgy_classifier_refactored import NaturgyIncidentClassifier

# Crear y entrenar clasificador
classifier = NaturgyIncidentClassifier()
results = classifier.train_pipeline('infomation.xlsx')

# Clasificar nueva incidencia
result = classifier.classify_incident(
    "Error al calcular factura CUPS ES0022000005514737AZ1P",
    additional_fields={
        'notas': 'No permite generar orden de cálculo',
        'tipo_ticket': 'Incident'
    }
)

print(f"Categoría: {result['type_info']['nombre']}")
print(f"Confianza: {result['confidence']:.3f}")
print(f"Criticidad: {result['type_info']['nivel_criticidad']}")
```

### Cargar Modelo Pre-entrenado

```python
import pickle

# Cargar modelo guardado
with open('outputs/models/naturgy_model_[timestamp].pkl', 'rb') as f:
    model_data = pickle.load(f)

classifier = model_data['classifier']
incident_types = model_data['incident_types']

# Usar directamente
prediction = classifier.classify_incident("Nueva incidencia...")
```

## 📊 Métricas y Evaluación

El sistema proporciona métricas completas:

- **Silhouette Score**: Calidad del clustering automático
- **Model Accuracy**: Precisión del modelo predictivo  
- **Cross-Validation**: Validación cruzada 5-fold
- **Confidence Distribution**: Distribución de confianzas
- **Category Balance**: Balanceado de tipos identificados

## 🔄 Integración con ITSM

### Ejemplo de Integración con Remedy

```python
class RemedyIntegration:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        self.classifier = self.model_data['classifier']
    
    def auto_triage_incident(self, incident_data):
        result = self.classifier.classify_incident(
            incident_data['summary'],
            {'tipo_ticket': incident_data['type']}
        )
        
        return {
            'assignment_group': result['type_info']['nombre'],
            'priority': self._calculate_priority(result),
            'confidence': result['confidence'],
            'auto_assignable': result['confidence'] > 0.8
        }
```

## 🧪 Testing y Validación

### Ejecutar Casos de Prueba

```bash
# Generar 100 casos de prueba
python3 src/test_classifier.py infomation.xlsx 100

# Generar casos personalizados
python3 src/test_classifier.py infomation.xlsx 50
```

### Métricas de los Casos de Prueba

- **Casos clasificados exitosamente**: Porcentaje de éxito
- **Distribución de confianza**: Alta/Media/Baja confianza
- **Casos de incertidumbre**: Requieren revisión manual
- **Distribución por categorías**: Balance de clasificaciones

## 📄 Documentación Adicional

- **Documentación Técnica**: `docs/README.md`
- **Resumen Ejecutivo**: `docs/resumen_ejecutivo_proyecto_naturgy.txt`  
- **Explicación Centroides**: `docs/explicacion_centroides_naturgy.txt`
- **Flujo de Datos**: `docs/flujo_datos_completo.txt`

## 🔧 Solución de Problemas

### Errores Comunes

1. **Error al cargar Excel**: Verificar que openpyxl esté instalado
2. **Memoria insuficiente**: Reducir `tfidf_max_features` en config
3. **Pocos datos**: Mínimo 500 registros para entrenamiento
4. **Columnas faltantes**: Verificar 'Ticket ID', 'Resumen', 'Notas'

### Optimización de Performance

- Para datasets grandes (>10k): Usar `max_clusters=30`
- Para análisis rápido: Reducir `tfidf_max_features=5000`
- Para alta precisión: Aumentar `cv_folds=10`

## 🚀 Comandos de Ejecución Rápida

```bash
# 1. Análisis completo estándar
python3 src/naturgy_classifier_refactored.py infomation.xlsx

# 2. Análisis con casos de prueba para validación  
python3 src/test_classifier.py infomation.xlsx

# 3. Análisis personalizado con nombre
python3 src/naturgy_classifier_refactored.py infomation.xlsx analisis_custom

# 4. Prueba de corrección de CUPS (nuevo)
python3 src/test_cups_correction.py
```

---

## 🏆 Logros del Proyecto

✅ **Sistema de IA completo** para clasificación automática de incidencias  
✅ **24 categorías técnicas** identificadas automáticamente  
✅ **Reglas semánticas avanzadas** con alta precisión  
✅ **Estructura organizada** de salida con reportes categorizados  
✅ **Sistema de confianza** con detección de incertidumbre  
✅ **Generación automática** de casos de prueba para validación  
✅ **Nomenclatura coherente** y semánticamente significativa  
✅ **API lista para producción** e integración con ITSM  

**Impacto**: Sistema que transforma el triaje de incidencias con clasificación automática de alta precisión, habilitando ahorro significativo de recursos y mejora en SLAs de resolución.

---

*Desarrollado para Naturgy - Sistema Delta de Gestión de Incidencias*  
*Versión Refactorizada - Octubre 2025*
