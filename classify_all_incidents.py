#!/usr/bin/env python3
"""
Script para clasificar TODAS las incidencias individualmente
y obtener estadísticas reales de reglas semánticas vs modelo predictivo
"""

import pandas as pd
import pickle
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Agregar el directorio src al path para importar el módulo
sys.path.append('src')

# Importar el módulo con todas las clases necesarias
try:
    from naturgy_classifier_refactored import (
        NaturgyIncidentClassifier, 
        TextPreprocessor, 
        EntityExtractor, 
        IncidentClusterer, 
        PredictiveClassifier,
        OutputManager,
        CategoryNamingEngine
    )
    print("✅ Todas las clases necesarias importadas")
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    sys.exit(1)

def load_trained_model(model_path):
    """Carga el modelo entrenado"""
    print(f"📦 Cargando modelo desde {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Modelo cargado exitosamente")
        return model_data['classifier'] if 'classifier' in model_data else model_data
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def classify_all_incidents(data_path, model_path):
    """Clasifica todas las incidencias individualmente"""
    
    # Cargar modelo entrenado
    classifier = load_trained_model(model_path)
    if not classifier:
        return None
    
    # Cargar datos
    print(f"📊 Cargando datos desde {data_path}...")
    try:
        df = pd.read_excel(data_path)
        print(f"✅ Datos cargados: {df.shape[0]} registros")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None
    
    # Verificar que el modelo esté entrenado
    if not hasattr(classifier, 'is_trained') or not classifier.is_trained:
        print("❌ El modelo no está entrenado")
        return None
    
    # Estadísticas para recolectar
    stats = {
        'total_incidencias': 0,
        'reglas_semanticas': 0,
        'modelo_predictivo': 0,
        'sin_determinar': 0,
        'errores': 0,
        'por_categoria_semantica': defaultdict(int),
        'por_categoria_predictiva': defaultdict(int),
        'detalles': []
    }
    
    print(f"🔍 Clasificando {df.shape[0]} incidencias individualmente...")
    
    # Procesar cada incidencia
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Procesando incidencia {idx + 1}/{df.shape[0]}...")
        
        try:
            # Preparar datos para clasificación
            resumen = str(row.get('Resumen', ''))
            additional_fields = {
                'notas': str(row.get('Notas', '')),
                'tipo_ticket': str(row.get('Tipo de ticket', ''))
            }
            
            # Clasificar incidencia
            result = classifier.classify_incident(resumen, additional_fields)
            
            stats['total_incidencias'] += 1
            
            # Determinar si fue clasificada por reglas semánticas o modelo predictivo
            predicted_type = result.get('predicted_type', '')
            confidence = result.get('confidence', 0.0)
            
            # Verificar si es resultado de reglas semánticas
            is_semantic_rule = False
            is_sin_determinar = False
            
            # Categorías conocidas de reglas semánticas
            semantic_categories = {
                'gestion_cups', 'montaje_desmontaje_equipos', 'errores_calculo_facturacion',
                'lecturas_mediciones', 'direcciones_datos_cliente', 'cambio_titularidad',
                'ofertas_contratacion', 'tarifas_productos', 'gestion_contratos',
                'bono_social_vulnerabilidad', 'rechazos_bloqueos', 'cobros_pagos',
                'batch_procesos_automaticos', 'extracciones_informes', 'telemedida_medicion_remota',
                'errores_xml_mensajeria', 'integraciones_externas', 'campanas_marketing',
                'plantillas_documentacion', 'consultas_soporte_funcional', 'gestion_usuarios',
                'gestiones_internas_administrativas', 'gestion_ofertas', 'sincronizacion_datos',
                'sin_determinar'
            }
            
            if predicted_type in semantic_categories:
                is_semantic_rule = True
                if predicted_type == 'sin_determinar':
                    is_sin_determinar = True
                    stats['sin_determinar'] += 1
                else:
                    stats['reglas_semanticas'] += 1
                    stats['por_categoria_semantica'][predicted_type] += 1
            else:
                # Es del modelo predictivo (tipo_XX o nombres de clustering)
                stats['modelo_predictivo'] += 1
                stats['por_categoria_predictiva'][predicted_type] += 1
            
            # Guardar detalle para casos interesantes
            if idx < 10 or is_semantic_rule:  # Primeros 10 casos + todos los de reglas semánticas
                detail = {
                    'idx': idx,
                    'ticket_id': str(row.get('Ticket ID', f'idx_{idx}')),
                    'resumen': resumen[:100] + '...' if len(resumen) > 100 else resumen,
                    'predicted_type': predicted_type,
                    'confidence': confidence,
                    'method': 'reglas_semanticas' if is_semantic_rule else 'modelo_predictivo',
                    'sin_determinar': is_sin_determinar
                }
                stats['detalles'].append(detail)
            
        except Exception as e:
            stats['errores'] += 1
            print(f"⚠️ Error en incidencia {idx}: {e}")
    
    return stats

def generate_report(stats, output_path):
    """Genera reporte con las estadísticas"""
    if not stats:
        return
    
    print(f"📄 Generando reporte en {output_path}...")
    
    total = stats['total_incidencias']
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 ESTADÍSTICAS REALES DE CLASIFICACIÓN - TODAS LAS INCIDENCIAS")
    report_lines.append("=" * 80)
    report_lines.append(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total de incidencias procesadas: {total:,}")
    report_lines.append("")
    
    # Estadísticas principales
    report_lines.append("🎯 DISTRIBUCIÓN POR MÉTODO DE CLASIFICACIÓN:")
    report_lines.append(f"• Reglas semánticas exitosas: {stats['reglas_semanticas']:,} ({stats['reglas_semanticas']/total*100:.1f}%)")
    report_lines.append(f"• Sin determinar (reglas semánticas): {stats['sin_determinar']:,} ({stats['sin_determinar']/total*100:.1f}%)")
    report_lines.append(f"• Modelo predictivo: {stats['modelo_predictivo']:,} ({stats['modelo_predictivo']/total*100:.1f}%)")
    report_lines.append(f"• Errores: {stats['errores']:,} ({stats['errores']/total*100:.1f}%)")
    report_lines.append("")
    
    # Total reglas semánticas (exitosas + sin determinar)
    total_semantic = stats['reglas_semanticas'] + stats['sin_determinar']
    report_lines.append("📋 RESUMEN:")
    report_lines.append(f"• TOTAL procesado por reglas semánticas: {total_semantic:,} ({total_semantic/total*100:.1f}%)")
    report_lines.append(f"• TOTAL que llegó al modelo predictivo: {stats['modelo_predictivo']:,} ({stats['modelo_predictivo']/total*100:.1f}%)")
    report_lines.append("")
    
    # Detalle por categorías semánticas
    if stats['por_categoria_semantica']:
        report_lines.append("🎯 DISTRIBUCIÓN POR CATEGORÍAS SEMÁNTICAS:")
        sorted_semantic = sorted(stats['por_categoria_semantica'].items(), key=lambda x: x[1], reverse=True)
        for categoria, count in sorted_semantic:
            percentage = count/total*100
            report_lines.append(f"• {categoria}: {count:,} casos ({percentage:.1f}%)")
        report_lines.append("")
    
    # Top categorías predictivas
    if stats['por_categoria_predictiva']:
        report_lines.append("🤖 TOP 10 CATEGORÍAS DEL MODELO PREDICTIVO:")
        sorted_predictive = sorted(stats['por_categoria_predictiva'].items(), key=lambda x: x[1], reverse=True)[:10]
        for categoria, count in sorted_predictive:
            percentage = count/total*100
            report_lines.append(f"• {categoria}: {count:,} casos ({percentage:.1f}%)")
        report_lines.append("")
    
    # Ejemplos
    report_lines.append("📋 EJEMPLOS DE CLASIFICACIÓN:")
    semantic_examples = [d for d in stats['detalles'] if d['method'] == 'reglas_semanticas'][:5]
    predictive_examples = [d for d in stats['detalles'] if d['method'] == 'modelo_predictivo'][:5]
    
    if semantic_examples:
        report_lines.append("\n🎯 Ejemplos de reglas semánticas:")
        for ex in semantic_examples:
            report_lines.append(f"• {ex['ticket_id']}: {ex['resumen']} → {ex['predicted_type']} (confianza: {ex['confidence']:.3f})")
    
    if predictive_examples:
        report_lines.append("\n🤖 Ejemplos de modelo predictivo:")
        for ex in predictive_examples:
            report_lines.append(f"• {ex['ticket_id']}: {ex['resumen']} → {ex['predicted_type']} (confianza: {ex['confidence']:.3f})")
    
    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("✅ Reporte generado exitosamente")
    print(f"\n📊 RESULTADOS RÁPIDOS:")
    print(f"Total reglas semánticas: {total_semantic:,} ({total_semantic/total*100:.1f}%)")
    print(f"Total modelo predictivo: {stats['modelo_predictivo']:,} ({stats['modelo_predictivo']/total*100:.1f}%)")

def main():
    """Función principal"""
    print("🚀 CLASIFICACIÓN INDIVIDUAL DE TODAS LAS INCIDENCIAS")
    print("=" * 60)
    
    # Rutas
    data_path = "infomation.xlsx"
    model_path = "outputs_completo/models/naturgy_model_20251028_130413.pkl"
    output_path = "estadisticas_clasificacion_completa.txt"
    
    # Verificar archivos
    if not Path(data_path).exists():
        print(f"❌ No se encuentra el archivo de datos: {data_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ No se encuentra el modelo: {model_path}")
        return
    
    # Ejecutar clasificación
    stats = classify_all_incidents(data_path, model_path)
    
    if stats:
        # Generar reporte
        generate_report(stats, output_path)
        
        # Guardar estadísticas en JSON
        json_path = "estadisticas_clasificacion_completa.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"📊 Estadísticas JSON guardadas en: {json_path}")
        
        print("\n" + "=" * 60)
        print("✅ ANÁLISIS COMPLETADO")
        print(f"📄 Reporte detallado: {output_path}")
    else:
        print("❌ No se pudieron obtener estadísticas")

if __name__ == "__main__":
    main()
