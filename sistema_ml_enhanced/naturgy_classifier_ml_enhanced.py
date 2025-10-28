#!/usr/bin/env python3
"""
Sistema de Clasificación Naturgy - Versión ML Enhanced
Versión que depende más del modelo predictivo y menos de reglas semánticas directas,
con nomenclatura contextual automática integrada.
"""

import pandas as pd
import numpy as np
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class ContextualNameGenerator:
    """Generador de nombres contextuales para categorías del clustering"""
    
    def __init__(self):
        self.mapeo_tecnico_empresarial = {
            # Tipos de problema
            'fail': 'fallos', 'error': 'errores', 'problema': 'problemas',
            'excessive': 'problemas', 'long': 'problemas', 'lentitud': 'problemas de rendimiento',
            'critico': 'críticos', 'frecuentes': '', 'masivo': '',
            
            # Áreas funcionales
            'factura': 'facturación', 'facturas': 'facturación', 'billing': 'facturación',
            'contrato': 'contratos', 'contratos': 'contratos', 'contract': 'contratos',
            'batch': 'procesos automáticos', 'job': 'procesos automáticos', 'jobs': 'procesos automáticos',
            'datos': 'gestión de datos', 'data': 'gestión de datos',
            'atlas': 'sistema', 'sistema': 'sistema', 'delta': 'sistema',
            'infraestructura': 'infraestructura', 'cups': 'gestión de CUPS',
            'aparato': 'equipos', 'equipos': 'equipos', 'montaje': 'equipos',
            'usuario': 'usuarios', 'usuarios': 'usuarios', 'user': 'usuarios',
            'consulta': 'consultas', 'consultas': 'consultas', 'query': 'consultas'
        }
    
    def generar_nombre_contextual(self, palabras_clave, ejemplos, descripcion, criticidad, nombre_original):
        """Genera un nombre contextual basado en análisis semántico mejorado"""
        
        # Combinar toda la información disponible
        texto_completo = ' '.join(palabras_clave + [descripcion]).lower()
        
        # Añadir ejemplos al análisis
        ejemplos_texto = ""
        if ejemplos:
            for ej in ejemplos[:3]:  # Usar hasta 3 ejemplos
                if isinstance(ej, dict) and 'Resumen' in ej:
                    ejemplos_texto += f" {ej['Resumen']}"
                elif isinstance(ej, dict) and 'resumen' in ej:
                    ejemplos_texto += f" {ej['resumen']}"
        
        texto_completo += ejemplos_texto.lower()
        
        # ANÁLISIS SEMÁNTICO MEJORADO
        
        # 1. Detectar TIPO DE PROBLEMA
        tipo_problema = "Gestión"  # Default
        if any(term in texto_completo for term in ['fail', 'error', 'fallo', 'problema']):
            tipo_problema = "Errores"
        elif any(term in texto_completo for term in ['excessive', 'long', 'lento', 'lentitud']):
            tipo_problema = "Problemas de Rendimiento"
        elif any(term in texto_completo for term in ['solicitud', 'request', 'peticion']):
            tipo_problema = "Solicitudes"
        elif any(term in texto_completo for term in ['consulta', 'query', 'prueba', 'test']):
            tipo_problema = "Consultas"
        
        # 2. Detectar ÁREA FUNCIONAL
        area_funcional = "Sistema"  # Default
        
        # Facturación
        if any(term in texto_completo for term in ['factura', 'facturas', 'billing', 'cobro', 'pago', 'importe', 'euro']):
            area_funcional = "Facturación"
            if criticidad == 'Alta':
                return f"Problemas Críticos de {area_funcional}"
        
        # Contratos
        elif any(term in texto_completo for term in ['contrato', 'contratos', 'alta', 'baja', 'modificar']):
            area_funcional = "Contratos"
        
        # CUPS
        elif any(term in texto_completo for term in ['cups', 'codigo', 'punto', 'suministro']):
            area_funcional = "CUPS"
        
        # Procesos Batch
        elif any(term in texto_completo for term in ['batch', 'job', 'proceso', 'd5', 'dectd', 'detf', 'running', 'cyclic']):
            area_funcional = "Procesos Batch"
            if tipo_problema == "Gestión":
                tipo_problema = "Fallos"
        
        # Datos
        elif any(term in texto_completo for term in ['datos', 'data', 'apartadas', 'regularizadoras']):
            area_funcional = "Datos"
        
        # Equipos
        elif any(term in texto_completo for term in ['aparato', 'equipo', 'montaje', 'desmontaje', 'renovacion']):
            area_funcional = "Equipos"
        
        # Infraestructura
        elif any(term in texto_completo for term in ['infraestructura', 'sistema', 'servicio', 'delta']):
            area_funcional = "Sistema"
        
        # Usuarios
        elif any(term in texto_completo for term in ['usuario', 'user', 'puesto', 'trabajo']):
            area_funcional = "Usuarios"
        
        # 3. REGLAS ESPECÍFICAS (más amplias)
        
        # Datos regulatorios (patrón muy específico)
        if 'apartadas' in texto_completo and 'regularizadoras' in texto_completo:
            return "Solicitudes de Datos Regulatorios"
        
        # Problemas de rendimiento
        if any(term in texto_completo for term in ['lentitud', 'lento', 'rendimiento', 'extrema']):
            return "Problemas de Rendimiento del Sistema"
        
        # Gestión CUPS (patrón común)
        if 'cups' in texto_completo:
            if any(term in texto_completo for term in ['regularizar', 'error', 'problema']):
                return "Errores en Gestión de CUPS"
            else:
                return "Gestión de CUPS"
        
        # Procesos batch fallidos
        if any(term in texto_completo for term in ['fail', 'excessive', 'long']) and any(term in texto_completo for term in ['batch', 'd5', 'dectd', 'detf']):
            return "Fallos en Procesos Batch"
        
        # 4. GENERACIÓN CON PATRÓN GRAMATICAL
        preposicion = "de" if area_funcional in ["Facturación", "CUPS", "Datos", "Equipos", "Sistema", "Usuarios"] else "en"
        
        # Casos especiales de nomenclatura
        if area_funcional == "CUPS":
            area_funcional = "Gestión de CUPS"
            preposicion = "en"
        elif area_funcional == "Datos":
            area_funcional = "Gestión de Datos"
            preposicion = "en"
        elif area_funcional == "Equipos":
            area_funcional = "Equipos y Renovaciones" 
            preposicion = "de"
        elif area_funcional == "Procesos Batch":
            preposicion = "en"
        
        # Construir nombre final
        if tipo_problema == "Gestión":
            nombre_final = f"Gestión {preposicion} {area_funcional}"
        else:
            nombre_final = f"{tipo_problema} {preposicion} {area_funcional}"
        
        # Limpiar y normalizar
        nombre_final = re.sub(r'\s+', ' ', nombre_final).strip()
        nombre_final = nombre_final.replace(" de de ", " de ").replace(" en en ", " en ")
        
        return nombre_final

class EnhancedSemanticAnalyzer:
    """Analizador semántico con umbrales más bajos para enviar más casos al ML"""
    
    def __init__(self):
        # Umbrales más restrictivos para depender menos de reglas semánticas
        self.confidence_threshold_high = 0.9  # Antes era 0.8
        self.confidence_threshold_medium = 0.7  # Antes era 0.6
        
        # Diccionarios semánticos (mantenemos los mismos pero con lógica más restrictiva)
        self.setup_semantic_dictionaries()
    
    def setup_semantic_dictionaries(self):
        """Configurar diccionarios semánticos"""
        self.semantic_categories = {
            'gestion_cups': {
                'keywords': ['cups', 'codigo', 'punto', 'suministro', 'identificador'],
                'patterns': [r'ES\d{4}\d{4}\d{4}\d{4}[A-Z]{2}', r'cups.*ES\d'],
                'weight': 1.0
            },
            'errores_calculo_facturacion': {
                'keywords': ['factura', 'facturación', 'cobro', 'importe', 'euro', 'precio', 'tarifa', 'coste'],
                'patterns': [r'factura.*\d+', r'importe.*\d+', r'€', r'euros?'],
                'weight': 1.0
            },
            'batch_procesos_automaticos': {
                'keywords': ['batch', 'proceso', 'automatico', 'job', 'programado', 'cron'],
                'patterns': [r'batch.*\d+', r'job.*\d+', r'proceso.*automatico'],
                'weight': 1.0
            }
            # ... otros diccionarios (mantener los existentes pero con peso reducido)
        }
    
    def analyze_incident(self, text, title=""):
        """Analiza una incidencia con umbrales más restrictivos"""
        if not text or pd.isna(text):
            return "sin_determinar", 0.0, "Texto vacío"
        
        text_lower = str(text).lower()
        combined_text = f"{title} {text}".lower()
        
        best_category = "sin_determinar"
        max_confidence = 0.0
        reasoning = "Sin coincidencias significativas"
        
        for category, config in self.semantic_categories.items():
            confidence = self._calculate_confidence(combined_text, config)
            
            # Umbrales más restrictivos
            if confidence > max_confidence and confidence >= self.confidence_threshold_high:
                max_confidence = confidence
                best_category = category
                reasoning = f"Coincidencia fuerte con {category}"
        
        # Solo clasificar si tenemos muy alta confianza, sino enviar al ML
        if max_confidence >= self.confidence_threshold_high:
            return best_category, max_confidence, reasoning
        else:
            return "sin_determinar", max_confidence, "Confianza insuficiente - enviar a ML"
    
    def _calculate_confidence(self, text, config):
        """Calcular confianza con criterios más estrictos"""
        confidence = 0.0
        
        # Contar keywords (requiere más coincidencias)
        keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text)
        keyword_confidence = (keyword_matches / len(config['keywords'])) * 0.6
        
        # Patrones (peso reducido)
        pattern_confidence = 0.0
        for pattern in config['patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_confidence = 0.3
                break
        
        confidence = (keyword_confidence + pattern_confidence) * config['weight']
        
        # Penalizar si el texto es muy corto o genérico
        if len(text) < 20:
            confidence *= 0.7
        
        return min(confidence, 1.0)

class MLEnhancedClassifier:
    """Clasificador híbrido que depende más del modelo predictivo"""
    
    def __init__(self):
        self.semantic_analyzer = EnhancedSemanticAnalyzer()
        self.name_generator = ContextualNameGenerator()
        self.vectorizer = None
        self.ml_model = None
        self.cluster_model = None
        self.category_definitions = {}
        self.is_trained = False
    
    def train(self, data_path, output_dir="ml_enhanced_outputs"):
        """Entrenar el clasificador híbrido"""
        print("🚀 Entrenando clasificador ML Enhanced...")
        
        # Preparar directorios
        output_path = Path(output_dir)
        for subdir in ['models', 'reports', 'data', 'logs']:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Cargar y preparar datos
        print("📊 Cargando datos...")
        df = pd.read_excel(data_path)
        
        # Preparar texto para análisis
        df['texto_completo'] = df.apply(
            lambda row: f"{row.get('Resumen', '')} {row.get('Descripción', '')} {row.get('Tipo de ticket', '')}",
            axis=1
        )
        
        # Preprocesar y vectorizar
        print("🔍 Vectorizando texto...")
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(df['texto_completo'].fillna(''))
        
        # Clustering (más clusters para mayor granularidad)
        print("🎯 Realizando clustering...")
        self.cluster_model = KMeans(n_clusters=30, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(X)
        
        # Generar definiciones de categorías con nombres contextuales
        print("📋 Generando definiciones contextuales...")
        self.category_definitions = self._generate_contextual_definitions(df, cluster_labels, X)
        
        # Entrenar modelo predictivo
        print("🤖 Entrenando modelo predictivo...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, cluster_labels, test_size=0.2, random_state=42
        )
        
        self.ml_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.ml_model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Modelo entrenado - Accuracy: {accuracy:.3f}")
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_path / 'models' / f'ml_enhanced_model_{timestamp}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'ml_model': self.ml_model,
                'cluster_model': self.cluster_model,
                'category_definitions': self.category_definitions,
                'semantic_analyzer': self.semantic_analyzer,
                'name_generator': self.name_generator
            }, f)
        
        print(f"💾 Modelo guardado: {model_path}")
        self.is_trained = True
        
        return accuracy, model_path
    
    def _generate_contextual_definitions(self, df, cluster_labels, X):
        """Generar definiciones con nombres contextuales automáticamente"""
        definitions = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_docs = df[cluster_mask]
            
            if len(cluster_docs) == 0:
                continue
            
            # Extraer características del cluster
            cluster_X = X[cluster_mask]
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Palabras más importantes del cluster
            if hasattr(cluster_X, 'toarray'):
                mean_scores = np.mean(cluster_X.toarray(), axis=0)
            else:
                mean_scores = np.mean(cluster_X, axis=0)
            
            top_indices = np.argsort(mean_scores)[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            
            # Ejemplos representativos
            ejemplos = cluster_docs.head(3).to_dict('records')
            
            # Determinar criticidad basada en tipo de ticket
            tipos_ticket = cluster_docs['Tipo de ticket'].value_counts()
            criticidad = "Alta" if "INCIDENCIA" in tipos_ticket.index else "Media"
            
            # Generar nombre técnico inicial
            nombre_tecnico = f"cluster_{cluster_id}"
            
            # Generar nombre contextual
            descripcion = f"Cluster {cluster_id} con {len(cluster_docs)} incidencias"
            nombre_contextual = self.name_generator.generar_nombre_contextual(
                top_keywords, ejemplos, descripcion, criticidad, nombre_tecnico
            )
            
            definitions[cluster_id] = {
                'nombre_original': nombre_tecnico,
                'nombre_contextual': nombre_contextual,
                'palabras_clave': top_keywords,
                'num_incidencias': len(cluster_docs),
                'ejemplos': ejemplos,
                'criticidad': criticidad,
                'tipos_principales': list(tipos_ticket.head(3).index)
            }
        
        return definitions
    
    def classify_incident(self, texto, titulo=""):
        """Clasificar una incidencia con el enfoque ML Enhanced"""
        if not self.is_trained:
            return {
                'categoria': 'sin_determinar',
                'confianza': 0.0,
                'metodo': 'no_entrenado',
                'razonamiento': 'Modelo no entrenado'
            }
        
        # Paso 1: Análisis semántico (con umbrales más altos)
        categoria_semantica, confianza_semantica, razonamiento = self.semantic_analyzer.analyze_incident(texto, titulo)
        
        # Solo usar reglas semánticas si tenemos MUY alta confianza
        if confianza_semantica >= 0.9:  # Umbral muy alto
            return {
                'categoria': categoria_semantica,
                'confianza': confianza_semantica,
                'metodo': 'reglas_semanticas',
                'razonamiento': razonamiento
            }
        
        # Paso 2: Clasificación ML (la mayoría de casos llegan aquí)
        try:
            texto_completo = f"{titulo} {texto}"
            X_text = self.vectorizer.transform([texto_completo])
            
            # Predicción del cluster
            cluster_pred = self.ml_model.predict(X_text)[0]
            cluster_proba = np.max(self.ml_model.predict_proba(X_text))
            
            # Obtener información del cluster
            if cluster_pred in self.category_definitions:
                categoria_info = self.category_definitions[cluster_pred]
                categoria_final = categoria_info['nombre_contextual']  # Usar nombre contextual
                
                return {
                    'categoria': categoria_final,
                    'confianza': float(cluster_proba),
                    'metodo': 'modelo_predictivo',
                    'razonamiento': f'Clasificado por ML en cluster {cluster_pred}',
                    'cluster_id': cluster_pred,
                    'palabras_clave': categoria_info['palabras_clave'][:5]
                }
            else:
                return {
                    'categoria': f'cluster_{cluster_pred}',
                    'confianza': float(cluster_proba),
                    'metodo': 'modelo_predictivo',
                    'razonamiento': 'Cluster sin definición contextual'
                }
        
        except Exception as e:
            return {
                'categoria': 'sin_determinar',
                'confianza': 0.0,
                'metodo': 'error',
                'razonamiento': f'Error en clasificación ML: {str(e)}'
            }

if __name__ == "__main__":
    # Ejemplo de uso
    classifier = MLEnhancedClassifier()
    print("🚀 Sistema de Clasificación ML Enhanced")
    print("Este sistema depende más del modelo predictivo y menos de reglas semánticas")
    print("Con nomenclatura contextual automática integrada")
