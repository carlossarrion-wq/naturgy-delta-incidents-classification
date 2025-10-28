#!/usr/bin/env python3
"""
Script para generar Excel con resultados de clasificación
Combina datos originales con categorías asignadas
"""

import pandas as pd
import json
import sys
from datetime import datetime
from pathlib import Path

def generar_excel_resultados(casos_originales_path, resultados_json_path, output_excel_path):
    """Genera Excel combinando datos originales con clasificaciones"""
    
    print("📊 GENERANDO EXCEL CON RESULTADOS DE CLASIFICACIÓN")
    print("=" * 60)
    
    try:
        # 1. Cargar datos originales
        print(f"📋 Cargando casos originales desde: {casos_originales_path}")
        df_originales = pd.read_excel(casos_originales_path)
        print(f"✅ Datos originales cargados: {len(df_originales)} registros")
        
        # 2. Cargar resultados de clasificación
        print(f"🔍 Cargando resultados desde: {resultados_json_path}")
        with open(resultados_json_path, 'r', encoding='utf-8') as f:
            resultados_data = json.load(f)
        
        casos_clasificados = resultados_data['casos_prueba']
        print(f"✅ Resultados cargados: {len(casos_clasificados)} casos clasificados")
        
        # 3. Crear diccionario de clasificaciones por Ticket ID
        clasificaciones = {}
        for caso in casos_clasificados:
            ticket_id = caso['ticket_id']
            clasificaciones[ticket_id] = {
                'categoria_predicha': caso['categoria_predicha'],
                'nombre_categoria': caso['nombre_categoria'],
                'confianza': caso['confianza'],
                'descripcion_categoria': caso['descripcion_categoria'],
                'criticidad': caso['criticidad'],
                'palabras_clave': ', '.join(caso.get('palabras_clave', [])),
                'resumen_clasificacion': caso['resumen_original']
            }
        
        # 4. Combinar datos originales con clasificaciones
        print("🔗 Combinando datos originales con clasificaciones...")
        
        # Agregar columnas de clasificación
        df_resultado = df_originales.copy()
        
        # Inicializar nuevas columnas
        df_resultado['Categoria_Predicha'] = ''
        df_resultado['Nombre_Categoria'] = ''
        df_resultado['Confianza'] = 0.0
        df_resultado['Descripcion_Categoria'] = ''
        df_resultado['Criticidad'] = ''
        df_resultado['Palabras_Clave'] = ''
        df_resultado['Estado_Clasificacion'] = 'No clasificado'
        
        # Rellenar datos de clasificación
        casos_encontrados = 0
        for idx, row in df_resultado.iterrows():
            ticket_id = str(row['Ticket ID'])
            
            if ticket_id in clasificaciones:
                clasificacion = clasificaciones[ticket_id]
                df_resultado.at[idx, 'Categoria_Predicha'] = clasificacion['categoria_predicha']
                df_resultado.at[idx, 'Nombre_Categoria'] = clasificacion['nombre_categoria']
                df_resultado.at[idx, 'Confianza'] = clasificacion['confianza']
                df_resultado.at[idx, 'Descripcion_Categoria'] = clasificacion['descripcion_categoria']
                df_resultado.at[idx, 'Criticidad'] = clasificacion['criticidad']
                df_resultado.at[idx, 'Palabras_Clave'] = clasificacion['palabras_clave']
                
                # Determinar estado basado en confianza
                if clasificacion['confianza'] < 0.70:
                    df_resultado.at[idx, 'Estado_Clasificacion'] = 'Sin determinar (< 0.70)'
                elif clasificacion['confianza'] <= 0.75:
                    df_resultado.at[idx, 'Estado_Clasificacion'] = 'Requiere revisión (≤ 0.75)'
                else:
                    df_resultado.at[idx, 'Estado_Clasificacion'] = 'Clasificado automáticamente'
                
                casos_encontrados += 1
        
        print(f"✅ Casos combinados exitosamente: {casos_encontrados} de {len(df_resultado)}")
        
        # 5. Reordenar columnas para mejor visualización
        print("📋 Organizando columnas...")
        
        # Columnas de clasificación al principio
        columnas_clasificacion = [
            'Ticket ID', 'Tipo de ticket', 'Estado', 'Resumen', 'Notas',
            'Categoria_Predicha', 'Nombre_Categoria', 'Confianza', 'Estado_Clasificacion',
            'Criticidad', 'Descripcion_Categoria', 'Palabras_Clave'
        ]
        
        # Resto de columnas originales
        otras_columnas = [col for col in df_resultado.columns if col not in columnas_clasificacion]
        
        # Reordenar
        columnas_finales = columnas_clasificacion + otras_columnas
        df_resultado = df_resultado[[col for col in columnas_finales if col in df_resultado.columns]]
        
        # 6. Crear múltiples hojas en el Excel
        print("📝 Generando Excel con múltiples hojas...")
        
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # Hoja principal con todos los datos
            df_resultado.to_excel(writer, sheet_name='Todos_los_Casos', index=False)
            
            # Hoja de resumen por categorías
            resumen_categorias = df_resultado[df_resultado['Nombre_Categoria'] != ''].groupby('Nombre_Categoria').agg({
                'Ticket ID': 'count',
                'Confianza': ['mean', 'min', 'max'],
                'Criticidad': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'No evaluada'
            }).round(3)
            
            resumen_categorias.columns = ['Total_Casos', 'Confianza_Promedio', 'Confianza_Min', 'Confianza_Max', 'Criticidad_Dominante']
            resumen_categorias = resumen_categorias.sort_values('Total_Casos', ascending=False)
            resumen_categorias.to_excel(writer, sheet_name='Resumen_por_Categorias')
            
            # Hoja de casos que requieren revisión
            casos_revision = df_resultado[
                (df_resultado['Confianza'] <= 0.75) & (df_resultado['Confianza'] > 0)
            ].sort_values('Confianza')
            
            casos_revision.to_excel(writer, sheet_name='Casos_Requieren_Revision', index=False)
            
            # Hoja de casos sin determinar
            casos_sin_determinar = df_resultado[df_resultado['Confianza'] < 0.70]
            casos_sin_determinar.to_excel(writer, sheet_name='Sin_Determinar', index=False)
            
            # Hoja de estadísticas generales
            estadisticas = pd.DataFrame({
                'Métrica': [
                    'Total de casos',
                    'Casos clasificados',
                    'Casos sin determinar (< 0.70)',
                    'Casos que requieren revisión (≤ 0.75)',
                    'Casos con clasificación automática',
                    'Confianza promedio',
                    'Categorías únicas identificadas'
                ],
                'Valor': [
                    len(df_resultado),
                    casos_encontrados,
                    len(casos_sin_determinar),
                    len(casos_revision),
                    len(df_resultado[(df_resultado['Confianza'] > 0.75)]),
                    df_resultado[df_resultado['Confianza'] > 0]['Confianza'].mean(),
                    len(df_resultado['Nombre_Categoria'].unique()) - 1  # -1 para excluir vacíos
                ]
            })
            
            estadisticas.to_excel(writer, sheet_name='Estadisticas_Generales', index=False)
        
        print(f"✅ Excel generado exitosamente: {output_excel_path}")
        
        # 7. Mostrar resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL EXCEL GENERADO")
        print("=" * 60)
        print(f"📁 Archivo: {output_excel_path}")
        print(f"📋 Total registros: {len(df_resultado)}")
        print(f"🎯 Casos clasificados: {casos_encontrados}")
        print(f"⚠️ Casos sin determinar (< 0.70): {len(casos_sin_determinar)}")
        print(f"🔍 Casos que requieren revisión (≤ 0.75): {len(casos_revision)}")
        print(f"✅ Casos con clasificación automática: {len(df_resultado[(df_resultado['Confianza'] > 0.75)])}")
        
        if casos_encontrados > 0:
            confianza_promedio = df_resultado[df_resultado['Confianza'] > 0]['Confianza'].mean()
            print(f"📈 Confianza promedio: {confianza_promedio:.3f}")
        
        print(f"🏷️ Categorías únicas: {len(df_resultado['Nombre_Categoria'].unique()) - 1}")
        
        print("\n📂 Hojas del Excel:")
        print("  • Todos_los_Casos: Datos completos con clasificaciones")
        print("  • Resumen_por_Categorias: Estadísticas por categoría")
        print("  • Casos_Requieren_Revision: Casos con confianza ≤ 0.75")
        print("  • Sin_Determinar: Casos con confianza < 0.70")
        print("  • Estadisticas_Generales: Métricas del análisis")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generando Excel: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if len(sys.argv) < 3:
        print("Uso: python generar_excel_resultados.py <casos_originales.xlsx> <resultados.json> [output.xlsx]")
        print("\nEjemplo:")
        print("  python generar_excel_resultados.py casos_prueba/data/casos_prueba_original.xlsx casos_prueba/data/casos_prueba_resultados.json resultados_clasificacion.xlsx")
        sys.exit(1)
    
    casos_originales_path = sys.argv[1]
    resultados_json_path = sys.argv[2]
    
    # Nombre de salida automático si no se especifica
    if len(sys.argv) > 3:
        output_excel_path = sys.argv[3]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel_path = f"resultados_clasificacion_{timestamp}.xlsx"
    
    success = generar_excel_resultados(casos_originales_path, resultados_json_path, output_excel_path)
    
    if not success:
        print("❌ El proceso falló. Revisa los errores anteriores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
