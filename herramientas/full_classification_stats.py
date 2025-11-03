#!/usr/bin/env python3
"""
Script para aplicar el proceso completo de clasificación (reglas semánticas + modelo)
a todas las incidencias y obtener solo los porcentajes por categoría
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, date
from collections import defaultdict
from pathlib import Path

# Agregar el directorio sistema_original al path
sys.path.append('sistema_original')

# Importar el clasificador
try:
    from naturgy_classifier_refactored import NaturgyIncidentClassifier
    print("✅ Clasificador importado correctamente")
except ImportError as e:
    print(f"❌ Error importando clasificador: {e}")
    sys.exit(1)

def train_classifier_for_rules(data_path):
    """Entrena un clasificador temporal solo para obtener las reglas semánticas"""
    print("🏋️ Entrenando clasificador temporal para acceder a reglas semánticas...")
    
    # Crear clasificador
    classifier = NaturgyIncidentClassifier(output_dir='temp_training')
    
    try:
        # Entrenar solo para tener acceso a classify_incident
        results = classifier.train_pipeline(data_path)
        print("✅ Clasificador entrenado - Reglas semánticas disponibles")
        
        # Limpiar archivos temporales
        if os.path.exists('temp_training'):
            import shutil
            shutil.rmtree('temp_training')
        
        return classifier
        
    except Exception as e:
        print(f"❌ Error entrenando clasificador: {e}")
        return None

def classify_all_and_count(data_path, nombre_prod_filter=None, fecha_desde=None, suffix=""):
    """Clasifica todas las incidencias y cuenta por categoría"""
    
    # Entrenar clasificador para tener reglas semánticas
    classifier = train_classifier_for_rules(data_path)
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
    
    # Aplicar filtros
    if nombre_prod_filter:
        df = df[df['Nombre Prod.'] == nombre_prod_filter].copy()
        print(f"🔍 Filtrado por Nombre Prod. = '{nombre_prod_filter}': {len(df)} registros")
    
    if fecha_desde:
        df['Fecha Creacion'] = pd.to_datetime(df['Fecha Creacion'])
        df = df[df['Fecha Creacion'] >= fecha_desde].copy()
        print(f"📅 Filtrado desde {fecha_desde}: {len(df)} registros")
    
    # Filtrar registros válidos
    df_valid = df[df['Resumen'].notna() & (df['Resumen'] != '')].copy()
    print(f"📋 Registros válidos para clasificación: {len(df_valid)}")
    
    # Contadores
    category_counts = defaultdict(int)
    method_counts = defaultdict(int)
    confidence_ranges = defaultdict(int)
    criticality_counts = defaultdict(int)
    total_processed = 0
    errors = 0
    
    # Categorías conocidas de reglas semánticas
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
    
    print(f"🔍 Aplicando clasificación completa a {len(df_valid)} incidencias...")
    print("⏳ Este proceso incluye reglas semánticas + modelo predictivo...")
    
    for idx, row in df_valid.iterrows():
        if idx % 500 == 0:
            print(f"   Procesadas {idx + 1}/{len(df_valid)} incidencias ({(idx+1)/len(df_valid)*100:.1f}%)...")
        
        try:
            # Preparar datos exactamente como en test_classifier.py
            resumen = str(row['Resumen']) if pd.notna(row['Resumen']) else ''
            additional_fields = {
                'notas': str(row.get('Notas', '')) if pd.notna(row.get('Notas', '')) else '',
                'tipo_ticket': str(row.get('Tipo de ticket', '')) if pd.notna(row.get('Tipo de ticket', '')) else ''
            }
            
            # Aplicar classify_incident (proceso completo: reglas semánticas + modelo)
            result = classifier.classify_incident(resumen, additional_fields)
            
            # Extraer información del resultado
            predicted_type = result.get('predicted_type', 'unknown')
            confidence = result.get('confidence', 0.0)
            type_info = result.get('type_info', {})
            category_name = type_info.get('nombre', predicted_type)
            criticality = type_info.get('nivel_criticidad', 'No evaluada')
            
            # Contar por categoría
            category_counts[category_name] += 1
            
            # Determinar método utilizado
            if predicted_type == 'sin_determinar':
                method_counts['sin_determinar'] += 1
            elif predicted_type in semantic_categories:
                method_counts['reglas_semanticas'] += 1
            else:
                # Cualquier otra cosa es del modelo predictivo
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
            if errors <= 3:  # Solo mostrar primeros 3 errores
                print(f"⚠️ Error en incidencia {idx}: {e}")
    
    # Resultados finales
    results = {
        'total_procesado': total_processed,
        'errores': errors,
        'por_categoria': dict(category_counts),
        'por_metodo': dict(method_counts),
        'por_confianza': dict(confidence_ranges),
        'por_criticidad': dict(criticality_counts)
    }
    
    print(f"✅ Clasificación completa finalizada:")
    print(f"   📊 Total procesado: {total_processed:,}")
    print(f"   ❌ Errores: {errors}")
    
    return results

def generate_final_report(stats, output_file='estadisticas_clasificacion_completa.txt'):
    """Genera reporte final con estadísticas de clasificación"""
    
    if not stats or stats['total_procesado'] == 0:
        print("❌ No hay datos válidos para generar reporte")
        return None
    
    print("📄 Generando reporte final de estadísticas...")
    
    total = stats['total_procesado']
    lines = []
    
    lines.append("=" * 85)
    lines.append("📊 ESTADÍSTICAS REALES - CLASIFICACIÓN COMPLETA DE TODAS LAS INCIDENCIAS")
    lines.append("=" * 85)
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total procesado: {total:,} incidencias")
    lines.append(f"Errores: {stats['errores']:,}")
    lines.append("")
    
    # Distribución por método de clasificación
    lines.append("🎯 DISTRIBUCIÓN POR MÉTODO DE CLASIFICACIÓN:")
    for method, count in sorted(stats['por_metodo'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        method_name = {
            'reglas_semanticas': 'Reglas semánticas exitosas',
            'sin_determinar': 'Sin determinar (reglas semánticas)',
            'modelo_predictivo': 'Modelo predictivo'
        }.get(method, method)
        lines.append(f"• {method_name}: {count:,} casos ({percentage:.1f}%)")
    
    # Resumen por método
    sem_exitosas = stats['por_metodo'].get('reglas_semanticas', 0)
    sin_det = stats['por_metodo'].get('sin_determinar', 0)
    sem_total = sem_exitosas + sin_det
    predictivo = stats['por_metodo'].get('modelo_predictivo', 0)
    
    lines.append("")
    lines.append("📋 RESUMEN POR MÉTODO:")
    lines.append(f"• TOTAL procesado por reglas semánticas: {sem_total:,} ({(sem_total/total)*100:.1f}%)")
    lines.append(f"  - Exitosas: {sem_exitosas:,} ({(sem_exitosas/total)*100:.1f}%)")
    lines.append(f"  - Sin determinar: {sin_det:,} ({(sin_det/total)*100:.1f}%)")
    lines.append(f"• TOTAL que llegó al modelo predictivo: {predictivo:,} ({(predictivo/total)*100:.1f}%)")
    lines.append("")
    
    # Top 20 categorías
    lines.append("📊 TOP 20 CATEGORÍAS MÁS FRECUENTES:")
    sorted_cats = sorted(stats['por_categoria'].items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (categoria, count) in enumerate(sorted_cats, 1):
        percentage = (count / total) * 100
        lines.append(f"{i:2d}. {categoria}: {count:,} casos ({percentage:.1f}%)")
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
    
    # Mostrar resumen en consola
    print("\n" + "🎯 RESULTADOS FINALES:")
    print(f"• Total procesado: {total:,} incidencias")
    print(f"• Reglas semánticas exitosas: {sem_exitosas:,} ({(sem_exitosas/total)*100:.1f}%)")
    print(f"• Sin determinar: {sin_det:,} ({(sin_det/total)*100:.1f}%)")
    print(f"• Modelo predictivo: {predictivo:,} ({(predictivo/total)*100:.1f}%)")
    print(f"• TOTAL reglas semánticas: {sem_total:,} ({(sem_total/total)*100:.1f}%)")
    
    return output_file

def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python full_classification_stats.py <archivo_datos.xlsx>")
        print("\nEjemplo:")
        print("  python full_classification_stats.py infomation.xlsx")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Verificar archivo
    if not Path(data_path).exists():
        print(f"❌ No se encuentra el archivo: {data_path}")
        sys.exit(1)
    
    print("🚀 INICIANDO CLASIFICACIÓN COMPLETA CON MÚLTIPLES FILTROS")
    print("🔄 Proceso: Reglas semánticas → Modelo predictivo (igual que test_classifier)")
    print("📊 Se generarán 3 reportes: General, DELTA SMILE, DELTA SMILE desde 2024")
    print("=" * 70)
    
    reportes_generados = []
    
    # 1. REPORTE GENERAL (sin filtros)
    print("\n" + "="*50)
    print("📊 REPORTE 1: TODAS LAS INCIDENCIAS")
    print("="*50)
    
    stats_general = classify_all_and_count(data_path)
    if stats_general:
        report_file = generate_final_report(stats_general, 'estadisticas_clasificacion_completa.txt')
        if report_file:
            reportes_generados.append(report_file)
            print(f"✅ Reporte general guardado: {report_file}")
    
    # 2. REPORTE DELTA SMILE (solo Nombre Prod. = DELTA SMILE)
    print("\n" + "="*50)
    print("📊 REPORTE 2: SOLO DELTA SMILE")
    print("="*50)
    
    stats_delta = classify_all_and_count(data_path, nombre_prod_filter="DELTA SMILE")
    if stats_delta:
        report_file = generate_final_report(stats_delta, 'estadisticas_clasificacion_DELTA_SMILE.txt')
        if report_file:
            reportes_generados.append(report_file)
            print(f"✅ Reporte DELTA SMILE guardado: {report_file}")
    
    # 3. REPORTE DELTA SMILE DESDE 2024 (Nombre Prod. = DELTA SMILE Y fecha >= 2024-01-01)
    print("\n" + "="*50)
    print("📊 REPORTE 3: DELTA SMILE DESDE ENERO 2024")
    print("="*50)
    
    fecha_2024 = datetime(2024, 1, 1)
    stats_delta_2024 = classify_all_and_count(
        data_path, 
        nombre_prod_filter="DELTA SMILE", 
        fecha_desde=fecha_2024
    )
    if stats_delta_2024:
        report_file = generate_final_report(stats_delta_2024, 'estadisticas_clasificacion_DELTA_SMILE_2024.txt')
        if report_file:
            reportes_generados.append(report_file)
            print(f"✅ Reporte DELTA SMILE 2024 guardado: {report_file}")
    
    # RESUMEN FINAL
    print("\n" + "="*70)
    print("🎉 PROCESO COMPLETADO - TODOS LOS REPORTES GENERADOS")
    print("="*70)
    
    if reportes_generados:
        print(f"📄 Total de reportes generados: {len(reportes_generados)}")
        for i, reporte in enumerate(reportes_generados, 1):
            print(f"   {i}. {reporte}")
        print("\n✅ TODOS LOS REPORTES DISPONIBLES")
    else:
        print("❌ No se pudieron generar los reportes")
        sys.exit(1)

if __name__ == "__main__":
    main()
