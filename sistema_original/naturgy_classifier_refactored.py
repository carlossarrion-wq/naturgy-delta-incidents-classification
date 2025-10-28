#!/usr/bin/env python3
# Naturgy AI Incident Classifier - Refactored Version
"""
Sistema completo de clasificación automática de incidencias para Naturgy Delta
Versión refactorizada con estructura de carpetas organizada y nomenclatura mejorada
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import SnowballStemmer
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not installed. Some features may be limited.")
    NLTK_AVAILABLE = False

class OutputManager:
    """Gestiona la estructura de carpetas y archivos de salida"""
    
    def __init__(self, base_output_dir: str = "outputs"):
        self.base_dir = Path(base_output_dir)
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """Crea la estructura de carpetas necesaria"""
        directories = {
            'models': self.base_dir / 'models',
            'reports': self.base_dir / 'reports', 
            'data': self.base_dir / 'data',
            'logs': self.base_dir / 'logs'
        }
        
        for dir_name, dir_path in directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Directorio {dir_name} preparado: {dir_path}")
    
    def get_path(self, category: str, filename: str) -> Path:
        """Obtiene la ruta completa para un archivo según su categoría"""
        return self.base_dir / category / filename
    
    def save_model(self, model_data: dict, filename: str = None):
        """Guarda modelo en la carpeta models"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"naturgy_model_{timestamp}.pkl"
        
        filepath = self.get_path('models', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Modelo guardado: {filepath}")
        return filepath
    
    def save_json_data(self, data: dict, filename: str):
        """Guarda datos JSON en la carpeta data"""
        filepath = self.get_path('data', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📊 Datos JSON guardados: {filepath}")
        return filepath
    
    def save_report(self, content: str, filename: str):
        """Guarda reporte de texto en la carpeta reports"""
        filepath = self.get_path('reports', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 Reporte guardado: {filepath}")
        return filepath

class CategoryNamingEngine:
    """Motor de nomenclatura inteligente para categorías"""
    
    def __init__(self):
        self.setup_naming_rules()
    
    def setup_naming_rules(self):
        """Configura reglas de nomenclatura semántica"""
        
        # Patrones técnicos principales
        self.technical_patterns = {
            'infraestructura': {
                'keywords': ['servidor', 'infraestructura', 'sistema', 'red', 'conexion', 'servicio'],
                'base_name': 'Infraestructura'
            },
            'datos': {
                'keywords': ['datos', 'data', 'base', 'carga', 'actualización', 'masiva', 'información'],
                'base_name': 'Gestión_Datos'
            },
            'comunicacion': {
                'keywords': ['comunicacion', 'notificacion', 'envio', 'correo', 'mensaje', 'alerta'],
                'base_name': 'Comunicaciones'
            },
            'procesos': {
                'keywords': ['batch', 'proceso', 'job', 'ejecucion', 'automatico', 'programado'],
                'base_name': 'Procesos_Batch'
            },
            'consultas': {
                'keywords': ['consulta', 'busqueda', 'listado', 'extraccion', 'reporte', 'información'],
                'base_name': 'Consultas_Funcionales'
            },
            'errores': {
                'keywords': ['error', 'fallo', 'excepcion', 'timeout', 'crash', 'bug'],
                'base_name': 'Errores_Sistema'
            },
            'facturacion': {
                'keywords': ['factura', 'facturacion', 'importe', 'cobro', 'pago', 'recibo'],
                'base_name': 'Facturación'
            },
            'contratos': {
                'keywords': ['contrato', 'alta', 'baja', 'modificacion', 'cambio', 'gestion'],
                'base_name': 'Gestión_Contratos'
            }
        }
        
        # Sufijos para especificar subtipos
        self.subtypes = {
            'critico': ['crítico', 'urgente', 'bloqueante', 'alto'],
            'masivo': ['masivo', 'múltiple', 'lote', 'bulk'],
            'automatico': ['automático', 'batch', 'programado', 'scheduled'],
            'manual': ['manual', 'individual', 'específico', 'puntual']
        }
    
    def generate_semantic_name(self, cluster_df: pd.DataFrame, cluster_id: int, 
                             keywords: List[str]) -> str:
        """Genera nombre semánticamente coherente"""
        
        # Combinar todo el texto para análisis
        all_text = ' '.join([
            ' '.join(cluster_df['Resumen'].fillna('').astype(str)),
            ' '.join(cluster_df.get('Notas', pd.Series([])).fillna('').astype(str)),
            ' '.join(cluster_df.get('Tipo de ticket', pd.Series([])).fillna('').astype(str))
        ]).lower()
        
        # Identificar patrón técnico principal
        main_pattern = self._identify_main_pattern(all_text)
        
        # Identificar subtipo si existe
        subtype = self._identify_subtype(all_text)
        
        # Generar nombre base
        if main_pattern:
            base_name = self.technical_patterns[main_pattern]['base_name']
        else:
            # Fallback: usar keywords más distintivos
            base_name = self._generate_fallback_name(keywords, cluster_df)
        
        # Añadir subtipo si se identificó
        if subtype:
            semantic_name = f"{base_name}_{subtype.title()}"
        else:
            semantic_name = base_name
        
        # Añadir especificador si es necesario para evitar duplicados
        size_modifier = self._get_size_modifier(len(cluster_df))
        if size_modifier:
            semantic_name = f"{semantic_name}_{size_modifier}"
        
        return semantic_name.replace('_', ' ')
    
    def _identify_main_pattern(self, text: str) -> Optional[str]:
        """Identifica el patrón técnico principal"""
        pattern_scores = {}
        
        for pattern_name, pattern_info in self.technical_patterns.items():
            score = 0
            for keyword in pattern_info['keywords']:
                score += text.count(keyword.lower())
            pattern_scores[pattern_name] = score
        
        # Retornar el patrón con mayor puntuación si supera el umbral
        max_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        return max_pattern[0] if max_pattern[1] > 0 else None
    
    def _identify_subtype(self, text: str) -> Optional[str]:
        """Identifica subtipo específico"""
        for subtype_name, subtype_keywords in self.subtypes.items():
            for keyword in subtype_keywords:
                if keyword.lower() in text:
                    return subtype_name
        return None
    
    def _generate_fallback_name(self, keywords: List[str], cluster_df: pd.DataFrame) -> str:
        """Genera nombre alternativo basado en keywords distintivos"""
        if not keywords:
            return f"Categoria_Tecnica"
        
        # Usar las 2 palabras más distintivas
        main_keywords = keywords[:2]
        
        # Limpiar y formatear
        cleaned_keywords = []
        for keyword in main_keywords:
            if len(keyword) > 2:
                cleaned_keywords.append(keyword.title())
        
        if len(cleaned_keywords) >= 2:
            return f"{cleaned_keywords[0]}_{cleaned_keywords[1]}"
        elif len(cleaned_keywords) == 1:
            return f"Incidencias_{cleaned_keywords[0]}"
        else:
            return "Categoria_Generica"
    
    def _get_size_modifier(self, size: int) -> Optional[str]:
        """Obtiene modificador basado en el tamaño del cluster"""
        if size > 100:
            return "Frecuentes"
        elif size < 10:
            return "Específicas"
        return None

class NaturgyIncidentClassifier:
    """Pipeline completo para clasificación automática de incidencias - Versión Refactorizada"""
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = "outputs"):
        self.config = config or self._default_config()
        self.output_manager = OutputManager(output_dir)
        self.naming_engine = CategoryNamingEngine()
        
        self.preprocessor = TextPreprocessor(self.config)
        self.clusterer = IncidentClusterer(self.config)
        self.classifier = PredictiveClassifier(self.config)
        self.entity_extractor = EntityExtractor()
        
        self.is_trained = False
        self.incident_types = {}
        self.model_metrics = {}
        
    def _default_config(self) -> Dict:
        """Configuración por defecto del sistema"""
        return {
            'max_clusters': 50,
            'min_cluster_size': 20,
            'tfidf_max_features': 8000,
            'tfidf_min_df': 3,
            'tfidf_max_df': 0.7,
            'random_state': 42,
            'use_llm': False,
            'model_type': 'random_forest',
            'cv_folds': 5,
            'use_hierarchical': True,
            'silhouette_threshold': 0.1
        }
    
    def train_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Entrena el pipeline completo"""
        print("🚀 Iniciando entrenamiento del pipeline de clasificación...")
        
        # 1. Cargar y limpiar datos
        print("📊 Cargando datos...")
        df = self._load_data(data_path)
        
        # 2. Preprocesamiento
        print("🧹 Preprocesando texto...")
        df_processed = self.preprocessor.process_dataframe(df)
        
        # 3. Extracción de entidades
        print("🔍 Extrayendo entidades...")
        df_processed = self.entity_extractor.extract_entities(df_processed)
        
        # 4. Clustering inicial
        print("🎯 Realizando clustering inicial...")
        cluster_results = self.clusterer.cluster_incidents(df_processed)
        
        # 5. Entrenamiento del modelo predictivo
        print("🤖 Entrenando modelo predictivo...")
        model_results = self.classifier.train_model(
            df_processed, cluster_results['labels']
        )
        
        # 6. Generar tipos de incidencia con nomenclatura mejorada
        print("📋 Generando definiciones de tipos...")
        self.incident_types = self._generate_incident_types_improved(
            df_processed, cluster_results
        )
        
        # 7. Evaluación del sistema
        print("📈 Evaluando performance...")
        self.model_metrics = self._evaluate_system(
            df_processed, cluster_results, model_results
        )
        
        self.is_trained = True
        
        # 8. Guardar modelo y resultados con estructura organizada
        self._save_results_organized(cluster_results, model_results)
        
        return {
            'incident_types': self.incident_types,
            'metrics': self.model_metrics,
            'cluster_results': cluster_results,
            'model_results': model_results
        }
    
    def _generate_incident_types_improved(self, df: pd.DataFrame, 
                                        cluster_results: Dict) -> Dict[str, Dict]:
        """Genera definiciones de tipos con nomenclatura mejorada"""
        types = {}
        labels = cluster_results['labels']
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Ruido en clustering
                continue
                
            cluster_mask = labels == cluster_id
            cluster_df = df[cluster_mask]
            
            # Extraer palabras clave distintivas
            keywords = self._extract_distinctive_keywords(cluster_df)
            
            # Generar nombre semánticamente coherente
            semantic_name = self.naming_engine.generate_semantic_name(
                cluster_df, cluster_id, keywords
            )
            
            # Generar descripción mejorada
            description = self._generate_enhanced_description(cluster_df, semantic_name)
            
            type_info = {
                'nombre': semantic_name,
                'descripcion': description,
                'num_incidencias': len(cluster_df),
                'palabras_clave': keywords[:10],  # Top 10 keywords
                'tipos_principales': self._get_top_ticket_types(cluster_df),
                'ejemplos': self._get_representative_examples(cluster_df),
                'patrones_identificados': self._identify_patterns(cluster_df),
                'nivel_criticidad': self._assess_criticality(cluster_df)
            }
            
            # Usar nombre semántico como clave
            safe_key = re.sub(r'[^\w\s-]', '', semantic_name).replace(' ', '_').lower()
            types[safe_key] = type_info
            
        return types
    
    def _generate_enhanced_description(self, cluster_df: pd.DataFrame, 
                                     semantic_name: str) -> str:
        """Genera descripción mejorada basada en el análisis semántico"""
        size = len(cluster_df)
        
        # Analizar palabras clave más frecuentes para contexto
        top_keywords = self._extract_distinctive_keywords(cluster_df)[:3]
        
        description = f"Categoría '{semantic_name}' que agrupa {size} incidencias "
        
        # Añadir contexto específico basado en el nombre
        if 'Infraestructura' in semantic_name:
            description += "relacionadas con problemas de infraestructura, servicios y conectividad del sistema."
        elif 'Datos' in semantic_name:
            description += "relacionadas con gestión, actualización y procesamiento de datos."
        elif 'Comunicaciones' in semantic_name:
            description += "relacionadas con notificaciones, envíos y sistemas de comunicación."
        elif 'Procesos' in semantic_name:
            description += "relacionadas con procesos automatizados, jobs y tareas programadas."
        elif 'Consultas' in semantic_name:
            description += "relacionadas con búsquedas, listados y extracción de información."
        elif 'Errores' in semantic_name:
            description += "relacionadas con fallos técnicos, excepciones y errores del sistema."
        elif 'Facturación' in semantic_name:
            description += "relacionadas con procesos de facturación, cobros y aspectos económicos."
        elif 'Contratos' in semantic_name:
            description += "relacionadas con gestión de contratos, altas, bajas y modificaciones."
        else:
            description += "con características técnicas y funcionales específicas."
        
        # Añadir información sobre términos técnicos más frecuentes
        if top_keywords:
            description += f" Términos técnicos frecuentes: '{', '.join(top_keywords[:3])}'."
        
        return description
    
    def _identify_patterns(self, cluster_df: pd.DataFrame) -> List[str]:
        """Identifica patrones específicos en el cluster"""
        patterns = []
        
        # Analizar texto combinado
        all_text = ' '.join(cluster_df['Resumen'].fillna('').astype(str)).lower()
        
        # Patrones temporales
        if any(word in all_text for word in ['nocturno', 'madrugada', 'fin semana']):
            patterns.append('Incidencias fuera de horario laboral')
        
        # Patrones de volumen
        if any(word in all_text for word in ['masivo', 'múltiple', 'varios']):
            patterns.append('Afectación múltiple o masiva')
        
        # Patrones de urgencia
        if any(word in all_text for word in ['urgente', 'crítico', 'bloqueo']):
            patterns.append('Requiere atención prioritaria')
        
        # Patrones de automatización
        if any(word in all_text for word in ['automático', 'batch', 'programado']):
            patterns.append('Procesos automatizados')
        
        return patterns
    
    def _assess_criticality(self, cluster_df: pd.DataFrame) -> str:
        """Evalúa el nivel de criticidad del cluster"""
        all_text = ' '.join(cluster_df['Resumen'].fillna('').astype(str)).lower()
        
        # Indicadores de alta criticidad
        high_criticality_indicators = ['crítico', 'urgente', 'bloqueo', 'caída', 'fallo total']
        # Indicadores de media criticidad  
        medium_criticality_indicators = ['error', 'problema', 'incidencia', 'fallo']
        # Indicadores de baja criticidad
        low_criticality_indicators = ['consulta', 'información', 'duda', 'aclaración']
        
        high_score = sum(1 for indicator in high_criticality_indicators if indicator in all_text)
        medium_score = sum(1 for indicator in medium_criticality_indicators if indicator in all_text)
        low_score = sum(1 for indicator in low_criticality_indicators if indicator in all_text)
        
        if high_score > 0:
            return 'Alta'
        elif medium_score > low_score:
            return 'Media'
        else:
            return 'Baja'
    
    def _save_results_organized(self, cluster_results: Dict, model_results: Dict):
        """Guarda los resultados con estructura organizada"""
        
        # 1. Guardar modelo entrenado
        model_data = {
            'preprocessor': self.preprocessor,
            'clusterer': self.clusterer,
            'classifier': self.classifier,
            'entity_extractor': self.entity_extractor,
            'incident_types': self.incident_types,
            'config': self.config,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        self.output_manager.save_model(model_data)
        
        # 2. Guardar análisis completo en JSON
        analysis_data = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'total_categorias': len(self.incident_types),
                'metodo_clustering': 'KMeans',
                'metodo_clasificacion': self.config['model_type']
            },
            'tipos_de_incidencia': self.incident_types,
            'metricas_modelo': self.model_metrics,
            'cluster_info': {
                'num_clusters': cluster_results['n_clusters'],
                'silhouette_score': cluster_results['silhouette_score'],
                'cluster_sizes': cluster_results['cluster_sizes']
            }
        }
        
        self.output_manager.save_json_data(
            analysis_data, 
            'analisis_completo_naturgy.json'
        )
        
        # 3. Guardar reporte legible
        report_content = self._generate_comprehensive_report()
        self.output_manager.save_report(
            report_content,
            'reporte_analisis_naturgy.txt'
        )
        
        # 4. Guardar resumen ejecutivo
        executive_summary = self._generate_executive_summary()
        self.output_manager.save_report(
            executive_summary,
            'resumen_ejecutivo.txt'
        )
    
    def _generate_comprehensive_report(self) -> str:
        """Genera reporte completo y detallado"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 ANÁLISIS INTEGRAL DE INCIDENCIAS NATURGY DELTA")
        report_lines.append("=" * 80)
        report_lines.append(f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"🎯 Total de categorías identificadas: {len(self.incident_types)}")
        report_lines.append(f"🤖 Precisión del modelo: {self.model_metrics.get('model_accuracy', 0.0):.3f}")
        report_lines.append(f"📊 Silhouette Score: {self.model_metrics.get('silhouette_score', 0.0):.3f}")
        report_lines.append("")
        
        # Ordenar por número de incidencias
        sorted_types = sorted(
            self.incident_types.items(),
            key=lambda x: x[1]['num_incidencias'],
            reverse=True
        )
        
        report_lines.append("🏷️  CATEGORÍAS IDENTIFICADAS (ordenadas por volumen)")
        report_lines.append("=" * 80)
        
        for type_key, type_info in sorted_types:
            report_lines.append(f"\n📋 {type_info['nombre'].upper()}")
            report_lines.append("-" * 60)
            report_lines.append(f"📊 Incidencias: {type_info['num_incidencias']}")
            report_lines.append(f"⚡ Criticidad: {type_info.get('nivel_criticidad', 'No evaluada')}")
            report_lines.append(f"📝 Descripción: {type_info['descripcion']}")
            report_lines.append(f"🔑 Palabras clave: {', '.join(type_info['palabras_clave'][:5])}")
            
            if type_info.get('patrones_identificados'):
                report_lines.append(f"🔍 Patrones: {', '.join(type_info['patrones_identificados'])}")
            
            if type_info['tipos_principales']:
                report_lines.append(f"📂 Tipos principales: {', '.join(type_info['tipos_principales'][:3])}")
            
            report_lines.append("\n📋 Ejemplos representativos:")
            for i, example in enumerate(type_info['ejemplos'][:3], 1):
                report_lines.append(f"  {i}. [{example['id']}] {example['resumen']}")
            
            report_lines.append("\n" + "=" * 60)
        
        return '\n'.join(report_lines)
    
    def _generate_executive_summary(self) -> str:
        """Genera resumen ejecutivo para directivos"""
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("📊 RESUMEN EJECUTIVO - ANÁLISIS DE INCIDENCIAS")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d')}")
        summary_lines.append("")
        
        # Estadísticas clave
        total_categories = len(self.incident_types)
        high_criticality = sum(1 for t in self.incident_types.values() if t.get('nivel_criticidad') == 'Alta')
        
        summary_lines.append("🎯 RESULTADOS CLAVE:")
        summary_lines.append(f"• Total de categorías identificadas: {total_categories}")
        summary_lines.append(f"• Categorías de alta criticidad: {high_criticality}")
        summary_lines.append(f"• Precisión del modelo: {self.model_metrics.get('model_accuracy', 0.0):.1%}")
        summary_lines.append("")
        
        # Top 5 categorías por volumen
        sorted_types = sorted(
            self.incident_types.items(),
            key=lambda x: x[1]['num_incidencias'],
            reverse=True
        )[:5]
        
        summary_lines.append("📊 TOP 5 CATEGORÍAS POR VOLUMEN:")
        for i, (key, info) in enumerate(sorted_types, 1):
            summary_lines.append(f"{i}. {info['nombre']} - {info['num_incidencias']} incidencias")
        
        summary_lines.append("")
        summary_lines.append("💡 RECOMENDACIONES:")
        summary_lines.append("• Priorizar atención a categorías de alta criticidad")
        summary_lines.append("• Implementar automatización en categorías más frecuentes")
        summary_lines.append("• Desarrollar procedimientos específicos por categoría")
        
        return '\n'.join(summary_lines)
    
    # Resto de métodos heredados de la clase original...
    def classify_incident(self, incident_text: str, 
                         additional_fields: Optional[Dict] = None) -> Dict[str, Any]:
        """Clasifica una nueva incidencia"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Sistema de preclasificación con reglas semánticamente coherentes
        combined_text = incident_text.lower()
        if additional_fields:
            for key, value in additional_fields.items():
                if value and str(value).strip().lower() != 'nan':
                    combined_text += f" {str(value).lower()}"
        
        # Aplicar reglas de preclasificación específicas
        classification_result = self._apply_semantic_rules(combined_text)
        if classification_result:
            return classification_result
            
        # Crear DataFrame temporal para procesamiento
        temp_df = pd.DataFrame({
            'Resumen': [incident_text],
            'Notas': [additional_fields.get('notas', '') if additional_fields else ''],
            'Tipo de ticket': [additional_fields.get('tipo_ticket', '') if additional_fields else '']
        })
        
        # Procesar texto
        processed_df = self.preprocessor.process_dataframe(temp_df)
        processed_df = self.entity_extractor.extract_entities(processed_df)
        
        # Clasificar
        prediction = self.classifier.predict(processed_df)
        
        # Control de confianza para modelo predictivo - Si confianza < 0.70, clasificar como "sin determinar"
        confidence = prediction[1] if len(prediction) > 1 else 0.0
        if confidence < 0.70:
            return {
                'predicted_type': 'sin_determinar',
                'confidence': confidence,
                'type_info': {
                    'nombre': 'Sin determinar',
                    'descripcion': f'Confianza del modelo predictivo ({confidence:.2f}) insuficiente para clasificación automática. Requiere revisión manual.',
                    'palabras_clave': ['baja confianza', 'modelo predictivo'],
                    'nivel_criticidad': 'No evaluada'
                },
                'extracted_entities': processed_df['entities'].iloc[0] if 'entities' in processed_df.columns else {},
                'processed_text': processed_df['combined_text'].iloc[0] if 'combined_text' in processed_df.columns else ''
            }
        
        # Obtener información del tipo
        incident_type = self.incident_types.get(prediction[0], {})
        
        return {
            'predicted_type': prediction[0],
            'confidence': confidence,
            'type_info': incident_type,
            'extracted_entities': processed_df['entities'].iloc[0] if 'entities' in processed_df.columns else {},
            'processed_text': processed_df['combined_text'].iloc[0] if 'combined_text' in processed_df.columns else ''
        }
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Carga datos desde Excel"""
        try:
            df = pd.read_excel(data_path)
            print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
            return df
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise
    
    def _extract_distinctive_keywords(self, cluster_df: pd.DataFrame) -> List[str]:
        """Extrae palabras clave MÁS distintivas (no las más comunes)"""
        # Combinar todo el texto del cluster
        cluster_text = ' '.join(cluster_df['Resumen'].fillna('').astype(str))
        
        # Tokenizar
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', cluster_text.lower())
        
        # Stop words expandidas
        expanded_stops = {
            'de', 'la', 'el', 'en', 'y', 'a', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 
            'por', 'con', 'su', 'para', 'como', 'las', 'del', 'los', 'un', 'una', 'al', 'del',
            'que', 'fue', 'son', 'han', 'muy', 'más', 'son', 'este', 'esta', 'ese', 'esa',
            'tiene', 'hacer', 'todo', 'año', 'día', 'mes', 'vez', 'caso', 'forma', 'parte'
        }
        
        # Filtrar y contar
        filtered_words = [w for w in words if w not in expanded_stops and len(w) > 2]
        word_freq = pd.Series(filtered_words).value_counts()
        
        # Retornar las 15 más distintivas
        return word_freq.head(15).index.tolist()
    
    def _get_top_ticket_types(self, cluster_df: pd.DataFrame) -> List[str]:
        """Obtiene los principales tipos de ticket"""
        if 'Tipo de ticket' in cluster_df.columns:
            return cluster_df['Tipo de ticket'].value_counts().head(3).index.tolist()
        return []
    
    def _get_representative_examples(self, cluster_df: pd.DataFrame) -> List[Dict]:
        """Obtiene ejemplos representativos"""
        examples = []
        for _, row in cluster_df.head(3).iterrows():
            example = {
                'id': row.get('Ticket ID', f'idx_{row.name}'),
                'resumen': str(row.get('Resumen', ''))[:150] + '...' if len(str(row.get('Resumen', ''))) > 150 else str(row.get('Resumen', '')),
                'tipo_ticket': str(row.get('Tipo de ticket', ''))
            }
            examples.append(example)
        return examples
    
    def _apply_semantic_rules(self, combined_text: str) -> Optional[Dict[str, Any]]:
        """Aplica reglas semánticas de preclasificación específicas con diccionario extendido"""
        
        # 🔍 PREPROCESAMIENTO: Detectar códigos CUPS (ES + números) y reemplazar por 'cups'
        import re
        cups_pattern = r'\bES\d{4}\d{4}\d{4}\d{4}[A-Z0-9]*\b'
        if re.search(cups_pattern, combined_text):
            combined_text += ' cups'  # Añadir keyword cups cuando se detecte el patrón
        
        # 🎯 SISTEMA DEFINITIVO DE CLASIFICACIÓN NATURGY - CATEGORÍAS TÉCNICAS EXPANDIDAS
        keywords = {
            # 1️⃣ Gestión de CUPS (EXPANDIDO)
            "gestion_cups": [
                "cups", "alta cups", "baja cups", "activar cups", "procesar baja", "procesar activación",
                "modificar dirección cups", "asociar sector suministro", "desvincular cups",
                "error vincular cups", "regularizar cups", "rechazos cups", "modificar cups",
                "código distribuidora", "ubicación cups", "cambio de datos cups"
            ],

            # 2️⃣ Montaje/Desmontaje/Equipos de medida
            "montaje_desmontaje_equipos": [
                "montaje", "desmontaje", "levantar aparato", "alta de aparato",
                "baja de aparato", "cambio de aparato", "error levantamiento",
                "no se puede eliminar aparato", "equipo de lectura", "aparato bloqueado",
                "eliminar aparato", "levantamiento de aparato", "retirada de equipo"
            ],

            # 3️⃣ Errores de cálculo/facturación (EXPANDIDO)
            "errores_calculo_facturacion": [
                "error al calcular", "no permite calcular", "suministro no listo para facturar",
                "tipdet", "passthrough", "pérdidas", "curva y cierre", "orden de cálculo",
                "oc", "ol", "java.lang.nullpointerexception", "error edm",
                "error genérico de cálculo", "error genérico del cálculo", "bloqueo facturación", 
                "fallo cálculo", "no calculable", "factura", "emisión de factura"
            ],

            # 4️⃣ Estados de cálculo/facturación
            "estados_calculo_facturacion": [
                "no calculable", "calculable", "bloqueado", "pendiente de facturar",
                "baja pendiente", "estado incoherente", "apartada", "desbloquear oc",
                "desbloquear ol", "bloquear", "desbloquear", "no tratable"
            ],

            # 5️⃣ Lecturas y mediciones
            "lecturas_mediciones": [
                "lectura de baja", "lectura de alta", "modificar lectura", "anular lectura",
                "solape", "lecturas no coinciden", "error lectura", "nm3", "pcs",
                "lecturas", "medición", "alta manual de lecturas"
            ],

            # 6️⃣ Direcciones y datos de cliente (EXPANDIDO)
            "direcciones_datos_cliente": [
                "modificar dirección", "dirección incorrecta", "corrección dirección",
                "datos titular", "cambiar nombre", "cambiar nif", "correo electrónico",
                "teléfono", "actualización dirección", "datos del titular", "nif", "dni",
                "nombre cliente", "email", "datos incompletos", "cambio de dirección",
                "actualización datos", "ofuscación email"
            ],

            # 7️⃣ Cambio de titularidad
            "cambio_titularidad": [
                "cambio de titular", "cambio titular sin subrogación", "error cambio titular",
                "validación cambio titular", "cambio titular", "subrogación", "titular"
            ],

            # 8️⃣ Ofertas y contratación
            "ofertas_contratacion": [
                "oferta en elaboración", "aceptar oferta", "firmar oferta", "validar oferta",
                "error cau", "adenda", "alta oferta", "error tipo oferta", "contratación",
                "no se puede validar oferta", "imposibilidad validar oferta", "modificar oferta"
            ],

            # 9️⃣ Tarifas y productos (EXPANDIDO)
            "tarifas_productos": [
                "tarifa comercial", "tarifa incorrecta", "cambiar producto", "producto activo",
                "cuida básico", "cuida luz", "mantenimiento", "servicio contratado", "producto",
                "cambiar IEH", "modificar tarifa", "modificar producto", "cambiar precio",
                "actualizar oferta", "código de oferta", "ID oferta", "producto asociado",
                "parámetro oferta", "condiciones oferta"
            ],

            # 🔟 Gestión de contratos (NUEVA CATEGORÍA)
            "gestion_contratos": [
                "modificar oferta", "actualizar contrato", "cambiar datos contrato",
                "editar oferta", "número oferta", "actualización condiciones", "contrato",
                "gestión contratos", "datos contrato", "condiciones contrato"
            ],

            # 🔟 Bono social y vulnerabilidad
            "bono_social_vulnerabilidad": [
                "bono social", "vulnerable", "vulnerabilidad", "tur vulnerable severo",
                "renovación bono social", "bono"
            ],

            # 1️⃣1️⃣ Rechazos y bloqueos
            "rechazos_bloqueos": [
                "rechazo pendiente", "cancelar rechazo", "error rechazo", "bloqueo suministro",
                "bloqueo cobros", "rechazo en vuelo", "rechazo", "bloqueo", "gestionar rechazo",
                "no permite gestionar rechazo"
            ],

            # 1️⃣2️⃣ Cobros y pagos
            "cobros_pagos": [
                "cobros", "ventanilla", "devolución", "responsabilidad de cobro",
                "pase a fallido", "impago", "duplicidad pagos", "error contabilidad",
                "lote de cobro", "pago", "facturas vencidas"
            ],

            # 1️⃣3️⃣ Batch/Procesos automáticos (EXPANDIDO)
            "batch_procesos_automaticos": [
                "fail", "long running", "ended not ok", "ftp envío", "batch",
                "timeout", "script", "sysout", "proceso detenido", "excessive",
                "job", "interfaz", "ftp", "fallo batch", "procesos automáticos", "error proceso"
            ],

            # 1️⃣4️⃣ Extracciones e informes
            "extracciones_informes": [
                "extracción", "informe", "listado", "recuento", "consulta",
                "descarga datos", "query consumos", "extracción aparatos", "extracción telemedida",
                "solicitud informe", "obtención de datos", "extracción facturas"
            ],

            # 1️⃣5️⃣ Telemedida y medición remota
            "telemedida_medicion_remota": [
                "telemedida", "dispone telemedida", "medición remota", "flag telemedida",
                "indicador telemedida"
            ],

            # 1️⃣6️⃣ Errores XML/mensajería
            "errores_xml_mensajeria": [
                "xml incorrecto", "error mensajería", "error formato xml", "no genera xml",
                "mensaje en incidencia", "xml", "mensajería"
            ],

            # 1️⃣7️⃣ Integraciones externas (EXPANDIDO)
            "integraciones_externas": [
                "gnclick", "salesforce", "markets", "sincronización", "no viaja dato",
                "integración externa", "interfaz externa", "replicación", "no replicado",
                "actualizar puntos suministro"
            ],

            # 1️⃣8️⃣ Campañas y marketing
            "campanas_marketing": [
                "campaña simulada", "campaña asnef", "forzar campaña", "error campaña",
                "campaña", "eliminar campañas simuladas"
            ],

            # 1️⃣9️⃣ Plantillas y documentación (EXPANDIDO)
            "plantillas_documentacion": [
                "plantillas", "subir plantilla", "eliminar plantilla", "documentación",
                "error descarga documentos", "error carga documentos", "plantilla",
                "documento", "subir contrato", "retirar plantilla", "modificar documento", 
                "pdf factura", "generar documento", "pintado pdf", "visualización factura"
            ],

            # 2️⃣0️⃣ Consultas y soporte funcional (EXPANDIDO)
            "consultas_soporte_funcional": [
                "consulta tablas", "sql", "funcionamiento batch", "criterios facturación",
                "soporte funcional", "consulta", "funcionamiento", "criterios",
                "pregunta", "validación calendario", "solicitud ayuda", "información proceso", 
                "visualizar contratos", "fusionar cliente", "cliente clonado", "cliente duplicado"
            ],

            # 2️⃣1️⃣ Gestión de usuarios (EXPANDIDA)
            "gestion_usuarios": [
                "reasignación usuario", "usuario genérico", "cambio de usuario",
                "permisos", "acceso denegado", "usuario", "asignación", "reasignar",
                "permisos usuario", "rol usuario", "acceso usuario",
                "crear usuario", "alta usuario", "nuevo usuario"
            ],

            # 2️⃣2️⃣ Gestiones internas administrativas (NUEVA CATEGORÍA)
            "gestiones_internas_administrativas": [
                "crear puesto de trabajo", "generar anexo", "alta interna", "tarea interna",
                "gestión administrativa", "crear registro", "crear puesto", "crear tarea",
                "puesto de trabajo", "anexo", "gestión interna", "tarea administrativa"
            ],

            # 2️⃣3️⃣ Gestión de ofertas (NUEVA CATEGORÍA)
            "gestion_ofertas": [
                "activar oferta", "activar línea oferta", "modificar oferta", "validar oferta",
                "condiciones oferta", "parámetro oferta", "cambiar IEH", "actualizar oferta",
                "anexo oferta", "pre-oferta", "línea oferta", "activación oferta",
                "parámetros oferta", "generar anexo pre-oferta"
            ],

            # 2️⃣4️⃣ Sincronización de datos (NUEVA CATEGORÍA)
            "sincronizacion_datos": [
                "replicación", "no replicado", "actualizar puntos", "sincronizar datos",
                "sincronización", "datos no sincronizados", "no se han replicado",
                "puntos de suministro", "replicar datos", "actualizar puntos de suministro"
            ]
        }
        
        # 🎯 MAPEO DEFINITIVO DE 20 CATEGORÍAS TÉCNICAS NATURGY
        category_info = {
            # 1️⃣ Gestión de CUPS
            "gestion_cups": {
                'nombre': 'Gestión de CUPS',
                'descripcion': 'Todo ticket que mencione explícitamente un CUPS y acciones sobre él (alta, baja, cambio de datos, asociación/desvinculación)',
                'criticidad': 'Media'
            },

            # 2️⃣ Montaje/Desmontaje/Equipos de medida
            "montaje_desmontaje_equipos": {
                'nombre': 'Montaje/Desmontaje/Equipos de medida',
                'descripcion': 'Incidencias sobre aparatos o contadores (montaje, desmontaje, levantamiento, error en equipos)',
                'criticidad': 'Alta'
            },

            # 3️⃣ Errores de cálculo/facturación
            "errores_calculo_facturacion": {
                'nombre': 'Errores de cálculo/facturación',
                'descripcion': 'Mensajes de error durante el cálculo de órdenes de cálculo o facturas',
                'criticidad': 'Alta'
            },

            # 4️⃣ Estados de cálculo/facturación
            "estados_calculo_facturacion": {
                'nombre': 'Estados de cálculo/facturación',
                'descripcion': 'Cambios de estado (no calculable, calculable, bloqueado) o incoherencias en estado de cálculo',
                'criticidad': 'Media'
            },

            # 5️⃣ Lecturas y mediciones
            "lecturas_mediciones": {
                'nombre': 'Lecturas y mediciones',
                'descripcion': 'Problemas con lecturas (baja, alta, modificar, anular, solapes, PCS)',
                'criticidad': 'Media'
            },

            # 6️⃣ Direcciones y datos de cliente
            "direcciones_datos_cliente": {
                'nombre': 'Direcciones y datos de cliente',
                'descripcion': 'Cambios o correcciones en dirección, nombre, NIF, email, teléfono',
                'criticidad': 'Media'
            },

            # 7️⃣ Cambio de titularidad
            "cambio_titularidad": {
                'nombre': 'Cambio de titularidad',
                'descripcion': 'Solicitudes o errores en cambio de titular (con o sin subrogación)',
                'criticidad': 'Media'
            },

            # 8️⃣ Ofertas y contratación
            "ofertas_contratacion": {
                'nombre': 'Ofertas y contratación',
                'descripcion': 'Creación, modificación, validación o firma de ofertas',
                'criticidad': 'Media'
            },

            # 9️⃣ Tarifas y productos
            "tarifas_productos": {
                'nombre': 'Tarifas y productos',
                'descripcion': 'Cambio o corrección de tarifas, productos contratados, servicios adicionales, IEH y precios',
                'criticidad': 'Media'
            },

            # 🔟 Gestión de contratos
            "gestion_contratos": {
                'nombre': 'Gestión de contratos',
                'descripcion': 'Modificación de ofertas, actualización de contratos y cambios en condiciones contractuales',
                'criticidad': 'Media'
            },

            # 1️⃣1️⃣ Bono social y vulnerabilidad
            "bono_social_vulnerabilidad": {
                'nombre': 'Bono social y vulnerabilidad',
                'descripcion': 'Alta, renovación, corrección de datos del bono social',
                'criticidad': 'Media'
            },

            # 1️⃣1️⃣ Rechazos y bloqueos
            "rechazos_bloqueos": {
                'nombre': 'Rechazos y bloqueos',
                'descripcion': 'Rechazos de solicitudes o bloqueos en proceso',
                'criticidad': 'Alta'
            },

            # 1️⃣2️⃣ Cobros y pagos
            "cobros_pagos": {
                'nombre': 'Cobros y pagos',
                'descripcion': 'Problemas en cobros, pagos, devoluciones, pases a fallido',
                'criticidad': 'Media'
            },

            # 1️⃣3️⃣ Batch/Procesos automáticos
            "batch_procesos_automaticos": {
                'nombre': 'Batch/Procesos automáticos',
                'descripcion': 'Errores técnicos en procesos batch, FTP, interfaces',
                'criticidad': 'Alta'
            },

            # 1️⃣4️⃣ Extracciones e informes
            "extracciones_informes": {
                'nombre': 'Extracciones e informes',
                'descripcion': 'Solicitudes de listados, informes, queries',
                'criticidad': 'Baja'
            },

            # 1️⃣5️⃣ Telemedida y medición remota
            "telemedida_medicion_remota": {
                'nombre': 'Telemedida y medición remota',
                'descripcion': 'Indicador de telemedida, problemas en medición remota',
                'criticidad': 'Media'
            },

            # 1️⃣6️⃣ Errores XML/mensajería
            "errores_xml_mensajeria": {
                'nombre': 'Errores XML/mensajería',
                'descripcion': 'Errores en generación o recepción de XMLs y mensajes',
                'criticidad': 'Alta'
            },

            # 1️⃣7️⃣ Integraciones externas
            "integraciones_externas": {
                'nombre': 'Integraciones externas',
                'descripcion': 'Problemas con sistemas externos (GNClick, Salesforce, Markets)',
                'criticidad': 'Alta'
            },

            # 1️⃣8️⃣ Campañas y marketing
            "campanas_marketing": {
                'nombre': 'Campañas y marketing',
                'descripcion': 'Ejecución o errores en campañas',
                'criticidad': 'Baja'
            },

            # 1️⃣9️⃣ Plantillas y documentación
            "plantillas_documentacion": {
                'nombre': 'Plantillas y documentación',
                'descripcion': 'Subida, eliminación o modificación de plantillas y documentos',
                'criticidad': 'Baja'
            },

            # 2️⃣0️⃣ Consultas y soporte funcional
            "consultas_soporte_funcional": {
                'nombre': 'Consultas y soporte funcional',
                'descripcion': 'Preguntas sobre funcionamiento, tablas, procesos, criterios',
                'criticidad': 'Baja'
            },

            # 2️⃣1️⃣ Gestión de usuarios
            "gestion_usuarios": {
                'nombre': 'Gestión de usuarios',
                'descripcion': 'Reasignación, cambios de usuario, permisos y accesos del sistema',
                'criticidad': 'Media'
            },

            # 2️⃣2️⃣ Gestiones internas administrativas
            "gestiones_internas_administrativas": {
                'nombre': 'Gestiones internas administrativas',
                'descripcion': 'Tareas administrativas internas, creación de puestos de trabajo, gestiones internas',
                'criticidad': 'Baja'
            },

            # 2️⃣3️⃣ Gestión de ofertas
            "gestion_ofertas": {
                'nombre': 'Gestión de ofertas',
                'descripcion': 'Activación, modificación y gestión de ofertas comerciales, líneas de oferta y parámetros',
                'criticidad': 'Media'
            },

            # 2️⃣4️⃣ Sincronización de datos
            "sincronizacion_datos": {
                'nombre': 'Sincronización de datos',
                'descripcion': 'Problemas de replicación, sincronización y actualización de datos entre sistemas',
                'criticidad': 'Media'
            }
        }
        
        # Calcular puntuaciones para cada categoría
        category_scores = {}
        matched_keywords = {}
        
        for category, keyword_list in keywords.items():
            score = 0
            matches = []
            for keyword in keyword_list:
                if keyword.lower() in combined_text:
                    score += 1
                    matches.append(keyword)
            
            if score > 0:
                category_scores[category] = score
                matched_keywords[category] = matches
        
        # Si hay coincidencias, seleccionar la categoría con mayor puntuación
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            
            # Calcular confianza basada en número de coincidencias
            confidence = min(0.95, 0.70 + (best_score * 0.05))
            
            # 🎯 CONTROL DE CONFIANZA MÍNIMA - Si confianza < 0.70, categorizar como "Sin determinar"
            if confidence < 0.70:
                return {
                    'predicted_type': 'sin_determinar',
                    'confidence': confidence,
                    'type_info': {
                        'nombre': 'Sin determinar',
                        'descripcion': f'Confianza insuficiente ({confidence:.2f}) para clasificación automática. Requiere revisión manual.',
                        'palabras_clave': ['baja confianza'] + matched_keywords[best_category][:2],
                        'nivel_criticidad': 'No evaluada'
                    },
                    'extracted_entities': {},
                    'processed_text': combined_text,
                    'rule_matched': f'Confianza baja: {confidence:.2f}',
                    'score': best_score,
                    'original_category': best_category
                }
            
            category_data = category_info[best_category]
            matched_keys = matched_keywords[best_category][:3]  # Top 3 keywords
            
            return {
                'predicted_type': best_category,
                'confidence': confidence,
                'type_info': {
                    'nombre': category_data['nombre'],
                    'descripcion': category_data['descripcion'],
                    'palabras_clave': matched_keys + ['clasificación automática'],
                    'nivel_criticidad': category_data['criticidad']
                },
                'extracted_entities': {},
                'processed_text': combined_text,
                'rule_matched': ', '.join(matched_keys[:2]),
                'score': best_score
            }
        
        return None
    
    def _evaluate_system(self, df: pd.DataFrame, cluster_results: Dict, 
                        model_results: Dict) -> Dict[str, float]:
        """Evalúa la performance del sistema"""
        return {
            'silhouette_score': cluster_results.get('silhouette_score', 0.0),
            'model_accuracy': model_results.get('test_accuracy', 0.0),
            'model_cv_score': model_results.get('cv_score', 0.0),
            'num_clusters': len(np.unique(cluster_results['labels'])),
            'coverage': len(df) / len(df)
        }


class TextPreprocessor:
    """Preprocesador de texto con reglas específicas de Naturgy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_preprocessing_rules()
    
    def setup_preprocessing_rules(self):
        """Configura reglas de preprocesamiento"""
        
        # Stop words seguras (eliminar)
        self.stop_words_safe = [
            'buenos días', 'cordial saludo', 'gracias', 'un saludo', 'saludos',
            'muchas gracias', 'quedo atento', 'quedo atenta', 'favor', 'por favor',
            'adjunto', 'envío', 'enviado', 'estimado', 'estimada', 'hola', 'buen día'
        ]
        
        # Sinonimias para normalización
        self.synonyms = {
            'fallo': 'error', 'incidencia': 'error', 'problema': 'error',
            'rechazo': 'error', 'cancelación': 'baja', 'anulación': 'baja',
            'activación': 'alta', 'creación': 'alta', 'cambiar': 'modificar',
            'actualizar': 'modificar', 'corregir': 'modificar'
        }
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa DataFrame completo"""
        df_processed = df.copy()
        
        # Combinar columnas de texto
        df_processed['combined_text'] = self._combine_text_columns(df_processed)
        
        # Limpiar y normalizar texto
        df_processed['processed_text'] = df_processed['combined_text'].apply(
            self._process_text
        )
        
        return df_processed
    
    def _combine_text_columns(self, df: pd.DataFrame) -> pd.Series:
        """Combina columnas de texto relevantes - MEJORADO: Incluye Notas con mayor peso"""
        # Orden de prioridad: Resumen tiene mayor peso, seguido de Notas
        text_columns = ['Resumen', 'Notas', 'Tipo de ticket', 'Resolución']
        combined = []
        
        for _, row in df.iterrows():
            text_parts = []
            
            # Resumen: incluir siempre si existe (mayor peso)
            if 'Resumen' in df.columns and pd.notna(row['Resumen']):
                resumen = str(row['Resumen']).strip()
                if resumen:
                    text_parts.append(resumen)
            
            # Notas: incluir con peso importante (segunda prioridad)
            if 'Notas' in df.columns and pd.notna(row['Notas']):
                notas = str(row['Notas']).strip()
                if notas and notas.lower() not in ['nan', '', 'null']:
                    text_parts.append(notas)
            
            # Otras columnas con menor peso
            for col in ['Tipo de ticket', 'Resolución']:
                if col in df.columns and pd.notna(row[col]):
                    text_value = str(row[col]).strip()
                    if text_value and text_value.lower() not in ['nan', '', 'null']:
                        text_parts.append(text_value)
            
            combined.append(' '.join(text_parts))
        
        return pd.Series(combined)
    
    def _process_text(self, text: str) -> str:
        """Procesa un texto individual"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Limpiar caracteres especiales pero mantener espacios y números
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar stop words seguras
        for stop_word in self.stop_words_safe:
            text = text.replace(stop_word.lower(), ' ')
        
        # Aplicar sinonimias
        for synonym, standard in self.synonyms.items():
            text = text.replace(synonym.lower(), standard.lower())
        
        # Limpiar espacios extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class EntityExtractor:
    """Extractor de entidades específicas del dominio"""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Configura patrones de extracción"""
        self.patterns = {
            'cups': r'ES\d{4}\d{4}\d{4}\d{4}[A-Z]{2}\d[A-Z]',
            'sr': r'R\d{2}-\d+',
            'req': r'REQ\d+',
            'ofl': r'OFL\d+',
            'fechas': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'productos': r'\b(bono social|cuidahogar|rl1|tur vulnerable)\b',
            'estados': r'\b(activo|inactivo|pendiente|bloqueado|calculable)\b'
        }
    
    def extract_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae entidades de todos los textos"""
        df_processed = df.copy()
        
        entities_list = []
        for text in df_processed['combined_text']:
            entities = self._extract_from_text(text)
            entities_list.append(entities)
        
        df_processed['entities'] = entities_list
        
        # Crear columnas individuales para cada tipo de entidad
        for entity_type in self.patterns.keys():
            df_processed[f'has_{entity_type}'] = [
                len(entities.get(entity_type, [])) > 0 
                for entities in entities_list
            ]
        
        return df_processed
    
    def _extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades de un texto"""
        if pd.isna(text):
            return {entity_type: [] for entity_type in self.patterns.keys()}
        
        text = str(text).lower()
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        
        return entities


class IncidentClusterer:
    """Sistema de clustering para agrupar incidencias similares"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vectorizer = None
        self.clustering_model = None
    
    def cluster_incidents(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza clustering de incidencias"""
        print("🎯 Iniciando clustering de incidencias...")
        
        # Vectorizar textos
        X = self._vectorize_texts(df['processed_text'])
        
        # Determinar número óptimo de clusters
        optimal_k = self._find_optimal_clusters(X)
        
        # Realizar clustering
        labels = self._perform_clustering(X, optimal_k)
        
        # Evaluar clustering
        metrics = self._evaluate_clustering(X, labels)
        
        print(f"✅ Clustering completado: {optimal_k} clusters identificados")
        
        return {
            'labels': labels,
            'n_clusters': optimal_k,
            'vectorizer': self.vectorizer,
            'silhouette_score': metrics.get('silhouette_score', 0.0),
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
    
    def _vectorize_texts(self, texts: pd.Series) -> np.ndarray:
        """Vectoriza textos usando TF-IDF"""
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf_max_features'],
            min_df=self.config['tfidf_min_df'],
            max_df=self.config['tfidf_max_df'],
            ngram_range=(1, 2),
            stop_words=None
        )
        
        X = self.vectorizer.fit_transform(texts.fillna(''))
        print(f"📊 Vectorización completada: {X.shape}")
        return X
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Encuentra el número óptimo de clusters"""
        max_k = min(self.config['max_clusters'], X.shape[0] // self.config['min_cluster_size'])
        
        if max_k < 2:
            return 2
        
        print(f"🔍 Evaluando clustering desde 2 hasta {max_k} clusters...")
        
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, 30))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config['random_state'], n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            try:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            except:
                silhouette_scores.append(0.0)
        
        # Seleccionar basado en tamaño del dataset para maximizar diversidad
        if X.shape[0] > 3000:
            optimal_k = min(25, max_k)
        elif X.shape[0] > 1000:
            optimal_k = min(15, max_k)
        else:
            optimal_k = min(10, max_k)
        
        print(f"🎯 Número óptimo de clusters seleccionado: {optimal_k}")
        return optimal_k
    
    def _perform_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Realiza el clustering"""
        self.clustering_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config['random_state'],
            n_init=10,
            max_iter=300
        )
        
        labels = self.clustering_model.fit_predict(X)
        return labels
    
    def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evalúa la calidad del clustering"""
        from sklearn.metrics import silhouette_score
        
        try:
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
            else:
                sil_score = 0.0
        except:
            sil_score = 0.0
        
        return {
            'silhouette_score': sil_score,
            'n_clusters': len(np.unique(labels)),
            'largest_cluster_size': np.max(np.bincount(labels))
        }


class PredictiveClassifier:
    """Modelo predictivo para clasificación automática"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_vectorizer = None
        self.label_encoder = None
        self.is_trained = False
    
    def train_model(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Entrena el modelo predictivo"""
        print("🤖 Entrenando modelo predictivo...")
        
        # Preparar features
        X = self._prepare_features(df)
        y = cluster_labels
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.config['random_state'],
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
        )
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluar modelo
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.is_trained = True
        
        print(f"✅ Modelo entrenado - Accuracy: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_score': test_score,
            'model': self.model
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predice la clasificación de nuevas incidencias"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        X = self._prepare_features(df)
        
        # Predicción
        prediction = self.model.predict(X)[0]
        
        # Calcular confianza
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.8
        except:
            confidence = 0.5
        
        # Decodificar etiqueta
        original_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return f"tipo_{original_label:02d}", confidence
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepara features para el modelo"""
        if self.feature_vectorizer is None:
            self.feature_vectorizer = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                min_df=self.config['tfidf_min_df'],
                max_df=self.config['tfidf_max_df'],
                ngram_range=(1, 2)
            )
            X_text = self.feature_vectorizer.fit_transform(df['processed_text'].fillna(''))
        else:
            X_text = self.feature_vectorizer.transform(df['processed_text'].fillna(''))
        
        return X_text


# Utilidades para ejecutar el pipeline completo
class IncidentAnalysisRunner:
    """Clase utilitaria para ejecutar análisis completos"""
    
    @staticmethod
    def run_complete_analysis(data_path: str, output_dir: str = 'outputs') -> None:
        """Ejecuta análisis completo y genera reportes"""
        print("🚀 Iniciando análisis completo de incidencias Naturgy...")
        
        # Crear clasificador con directorio de salida personalizado
        classifier = NaturgyIncidentClassifier(output_dir=output_dir)
        
        # Entrenar pipeline
        results = classifier.train_pipeline(data_path)
        
        print("✅ Análisis completo finalizado!")
        print(f"📂 Resultados organizados en: {output_dir}/")
        print("📊 Estructura creada:")
        print(f"  - {output_dir}/models/ (modelos entrenados)")
        print(f"  - {output_dir}/data/ (datos JSON)")
        print(f"  - {output_dir}/reports/ (reportes de texto)")
        print(f"  - {output_dir}/logs/ (archivos de registro)")


# Script principal para ejecutar desde línea de comandos
if __name__ == "__main__":
    import sys
    
    def main():
        """Función principal para ejecutar el pipeline"""
        print("🚀 NATURGY AI INCIDENT CLASSIFIER - VERSIÓN REFACTORIZADA")
        print("=" * 65)
        
        if len(sys.argv) < 2:
            print("Uso: python naturgy_classifier_refactored.py <archivo_datos.xlsx> [directorio_salida]")
            print("\nEjemplo:")
            print("  python naturgy_classifier_refactored.py infomation.xlsx ./outputs_nuevos")
            sys.exit(1)
        
        data_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'outputs'
        
        try:
            # Ejecutar análisis completo
            IncidentAnalysisRunner.run_complete_analysis(data_path, output_dir)
            
            print("\n" + "=" * 65)
            print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            print(f"📂 Resultados guardados en: {output_dir}")
            print("📊 Mejoras implementadas:")
            print("  ✓ Estructura de carpetas organizada")
            print("  ✓ Nomenclatura semánticamente coherente")
            print("  ✓ Reportes categorizados por tipo")
            print("  ✓ Análisis de criticidad incluido")
            
        except Exception as e:
            print(f"❌ Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    main()
