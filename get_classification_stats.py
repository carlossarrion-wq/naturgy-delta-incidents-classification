#!/usr/bin/env python3
"""
Script ligero para obtener SOLO los porcentajes de clasificación
aplicando el mismo proceso que los casos de prueba a todas las incidencias
"""

import pandas as pd
import pickle
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Agregar el directorio src al path
sys.path.append('src')

# Importar todas las clases necesarias para que pickle pueda deserializar
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
    print("✅ Clases necesarias importadas para deserialización")
except ImportError as e:
    print(f"❌ Error importando clases: {e}")
    sys.exit(1)

def load_classifier(model_path):
    """Carga el modelo entrenado"""
    print(f"📦 Cargando modelo desde {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # El modelo guardado puede tener diferentes estructuras
        if hasattr(model_data, 'classify_incident'):
            classifier = model_data
        elif 'classifier' in model_data:
            classifier = model_data['classifier']  
        else:
            print("❌ No se encontró clasificador válido en el modelo")
            return None
            
        if not hasattr(classifier, 'is_trained') or not classifier.is_trained:
            print("❌ El modelo no está entrenado")
            return None
            
        print("✅ Modelo cargado exitosamente")
        return classifier
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def classify_and_count(data_path, model_path):
    """Clasifica todas las incidencias y cuenta por categoría"""
    
    # Cargar modelo
    classifier = load_classifier(model_path)
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
    
    # Filtrar registros válidos
    df_valid = df[df['Resumen'].notna() & (df['Resumen'] != '')].copy()
    print(f"📋 Registros válidos: {len(df_valid)}")
    
    # Contadores
    category_counts = defaultdict(int)
    method_counts = defaultdict(int)
    confidence_ranges = defaultdict(int)
    criticality_counts = defaultdict(int)
    total_processed = 0
    errors = 0
    
    # Categorías de reglas semánticas conocidas
    semantic_categories = {
        'gestion_cups', 'montaje_desmontaje_equipos', 'errores_calculo_facturacion',
        'lecturas_mediciones', 'direcciones_datos_cliente', 'cambio_titularidad',
        'ofertas_contratacion', 'tarifas_productos', 'gestion_contratos',
        'bono_social_vulnerabilidad', 'rechazos_bloqueos', 'cobros_pagos',
        'batch_procesos_automaticos', 'extracciones_informes', 'telemedida_medicion_remota',
        'errores_xml_mensajeria', 'integraciones_externas', 'campanas_marketing',
        'plantillas_documentacion', 'consultas_soporte_funcional', 'gestion_usuarios',
        'gestiones_internas_administrativas', 'gestion_ofertas', 'sincronizacion_datos'
    }
    
    print(f"🔍 Clasificando {len(df_valid)} incidencias...")
    print("⏳ Solo contando por categorías (proceso más rápido)...")
    
    for idx, row in df_valid.iterrows():
        if idx % 1000 == 0:
            print(f"   Procesadas {idx + 1}/{len(df_valid)} incidencias...")
        
        try:
            # Preparar datos igual que en test_classifier.py
            resumen = str(row['Resumen']) if pd.notna(row['Resumen']) else ''
            additional_fields = {
                'notas': str(row.get('Notas', '')) if pd.notna(row.get('Notas', '')) else '',
                'tipo_ticket': str(row.get('Tipo de ticket', '')) if pd.notna(row.get('Tipo de ticket', '')) else ''
            }
            
            # Clasificar
            result = classifier.classify_incident(resumen, additional_fields)
            
            # Extraer información
            predicted_type = result.get('predicted_type', 'unknown')
            confidence = result.get('confidence', 0.0)
            type_info = result.get('type_info', {})
            category_name = type_info.get('nombre', predicted_type)
            criticality = type_info.get('nivel_criticidad', 'No evaluada')
            
            # Contar por categoría
            category_counts[category_name] += 1
            
            # Determinar método
            if predicted_type == 'sin_determinar':
                method_counts['sin_determinar'] += 1
            elif predicted_type in semantic_categories:
                method_counts['reglas_semanticas'] += 1
            else:
                method_counts['modelo_predictivo'] += 1
            
            # Contar por rango de confianza
            if confidence >= 0.8:
                confidence_ranges['alta_confianza'] += 1
            elif confidence >= 0.6:
                confidence_ranges['media_confianza'] += 1
            else:
                confidence_ranges['baja_confianza'] += 1
            
            # Contar por criticidad
            criticality_counts[criticality] += 1
            
            total_processed += 1
            
        except Exception as e:
            errors += 1
            if errors <= 5:  # Solo mostrar primeros 5 errores
                print(f"⚠️ Error en incidencia {idx}: {e}")
    
    # Resultados
    results = {
        'total_procesado': total_processed,
        'errores': errors,
        'por_categoria': dict(category_counts),
        'por_metodo': dict(method_counts),
        'por_confianza': dict(confidence_ranges),
        'por_criticidad': dict(criticality_counts)
    }
    
    print(f"✅ Clasificación completada:")
    print(f"   📊 Total procesado: {total_processed}")
    print(f"   ❌ Errores: {errors}")
    
    return results

def generate_report(stats, output_file='estadisticas_clasificacion.txt'):
    """Genera reporte con solo los porcentajes"""
    
    print("📄 Generando reporte de estadísticas...")
    
    total = stats['total_procesado']
    lines = []
    
    lines.append("=" * 80)
    lines.append("📊 ESTADÍSTICAS DE CLASIFICACIÓN - TODAS LAS INCIDENCIAS")
    lines.append("=" * 80)
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total procesado: {total:,} incidencias")
    lines.append(f"Errores: {stats['errores']:,}")
    lines.append("")
    
    # Distribución por método
    lines.append("🎯 DISTRIBUCIÓN POR MÉTODO DE CLASIFICACIÓN:")
    for method, count in sorted(stats['por_metodo'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        method_name = {
            'reglas_semanticas': 'Reglas semánticas exitosas',
            'sin_determinar': 'Sin determinar (reglas semánticas)', 
            'modelo_predictivo': 'Modelo predictivo'
        }.get(method, method)
        lines.append(f"• {method_name}: {count:,} casos ({percentage:.1f}%)")
    
    # Total reglas semánticas
    sem_exitosas = stats['por_metodo'].get('reglas_semanticas', 0)
    sin_det = stats['por_metodo'].get('sin_determinar', 0)
    sem_total = sem_exitosas + sin_det
    lines.append("")
    lines.append("📋 RESUMEN POR MÉTODO:")
    lines.append(f"• TOTAL procesado por reglas semánticas: {sem_total:,} ({(sem_total/total)*100:.1f}%)")
    lines.append(f"• TOTAL que llegó al modelo predictivo: {stats['por_metodo'].get('modelo_predictivo', 0):,} ({(stats['por_metodo'].get('modelo_predictivo', 0)/total)*100:.1f}%)")
    lines.append("")
    
    # Top categorías
    lines.append("📊 DISTRIBUCIÓN POR CATEGORÍAS (TOP 20):")
    sorted_cats = sorted(stats['por_categoria'].items(), key=lambda x: x[1], reverse=True)[:20]
    for categoria, count in sorted_cats:
        percentage = (count / total) * 100
        lines.append(f"• {categoria}: {count:,} casos ({percentage:.1f}%)")
    lines.append("")
    
    # Distribución por criticidad
    lines.append("⚡ DISTRIBUCIÓN POR CRITICIDAD:")
    for criticidad, count in sorted(stats['por_criticidad'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        lines.append(f"• {criticidad}: {count:,} casos ({percentage:.1f}%)")
    lines.append("")
    
    # Distribución por confianza
    lines.append("📈 DISTRIBUCIÓN POR NIVEL DE CONFIANZA:")
    for conf_level, count in sorted(stats['por_confianza'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        conf_name = {
            'alta_confianza': 'Alta confianza (≥0.8)',
            'media_confianza': 'Media confianza (0.6-0.8)',
            'baja_confianza': 'Baja confianza (<0.6)'
        }.get(conf_level, conf_level)
        lines.append(f"• {conf_name}: {count:,} casos ({percentage:.1f}%)")
    
    # Guardar archivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Reporte guardado: {output_file}")
    
    # También mostrar resumen en consola
    print("\n" + "🎯 RESULTADOS PRINCIPALES:")
    print(f"• Total procesado: {total:,} incidencias")
    print(f"• Reglas semánticas exitosas: {sem_exitosas:,} ({(sem_exitosas/total)*100:.1f}%)")
    print(f"• Sin determinar: {sin_det:,} ({(sin_det/total)*100:.1f}%)")
    print(f"• Modelo predictivo: {stats['por_metodo'].get('modelo_predictivo', 0):,} ({(stats['por_metodo'].get('modelo_predictivo', 0)/total)*100:.1f}%)")
    print(f"• Total reglas semánticas: {sem_total:,} ({(sem_total/total)*100:.1f}%)")
    
    return output_file

def main():
    """Función principal"""
    if len(sys.argv) < 3:
        print("Uso: python get_classification_stats.py <archivo_datos.xlsx> <modelo.pkl>")
        print("\nEjemplo:")
        print("  python get_classification_stats.py infomation.xlsx outputs_completo/models/naturgy_model_20251028_130413.pkl")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Verificar archivos
    if not Path(data_path).exists():
        print(f"❌ No se encuentra el archivo: {data_path}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"❌ No se encuentra el modelo: {model_path}")
        sys.exit(1)
    
    print("🚀 OBTENIENDO ESTADÍSTICAS DE CLASIFICACIÓN")
    print("=" * 50)
    
    # Clasificar y contar
    stats = classify_and_count(data_path, model_path)
    
    if not stats:
        print("❌ No se pudieron obtener estadísticas")
        sys.exit(1)
    
    # Generar reporte
    report_file = generate_report(stats)
    
    print(f"\n✅ PROCESO COMPLETADO")
    print(f"📄 Estadísticas guardadas en: {report_file}")

if __name__ == "__main__":
    main()
