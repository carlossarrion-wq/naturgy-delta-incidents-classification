#!/usr/bin/env python3
"""
Script para mejorar los nombres de las categorías del clustering
usando análisis contextual y reglas gramaticales.
"""

import json
import re
from collections import Counter

def extraer_conceptos_principales(palabras_clave, ejemplos, descripcion):
    """Extrae los conceptos más relevantes de una categoría"""
    
    # Diccionario de mapeo de términos técnicos a empresariales
    mapeo_tecnico_empresarial = {
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
    
    # Combinar todas las fuentes de información
    texto_completo = ' '.join(palabras_clave + [descripcion])
    texto_completo = texto_completo.lower()
    
    # Extraer conceptos clave
    conceptos = []
    for termino_original, termino_empresarial in mapeo_tecnico_empresarial.items():
        if termino_original in texto_completo and termino_empresarial:
            conceptos.append(termino_empresarial)
    
    return list(set(conceptos))  # Eliminar duplicados

def generar_nombre_contextual(categoria_info):
    """Genera un nombre contextual para una categoría"""
    
    palabras_clave = categoria_info.get('palabras_clave', [])
    ejemplos = [ej.get('resumen', '') for ej in categoria_info.get('ejemplos', [])]
    descripcion = categoria_info.get('descripcion', '')
    criticidad = categoria_info.get('nivel_criticidad', 'Media')
    nombre_actual = categoria_info.get('nombre', '')
    
    # Extraer conceptos principales
    conceptos = extraer_conceptos_principales(palabras_clave, ejemplos, descripcion)
    
    # Reglas específicas basadas en análisis del contenido
    texto_completo = ' '.join(palabras_clave + [descripcion]).lower()
    
    # Patrones específicos identificados en los datos
    if any(term in texto_completo for term in ['fail', 'excessive', 'long', 'running', 'cyclic']):
        if 'batch' in texto_completo or any(code in texto_completo for code in ['d5', 'dectd', 'detf']):
            return "Fallos en Procesos Batch"
        elif 'datos' in texto_completo or 'data' in texto_completo:
            return "Errores de Procesamiento de Datos"
    
    if 'factura' in texto_completo or 'billing' in texto_completo:
        if criticidad == 'Alta' or 'critico' in nombre_actual.lower():
            return "Problemas Críticos de Facturación"
        else:
            return "Consultas de Facturación"
    
    if 'contrato' in texto_completo or 'contract' in texto_completo:
        if 'error' in texto_completo or 'problema' in texto_completo:
            if 'automatico' in nombre_actual.lower():
                return "Errores en Gestión de Contratos"
            elif criticidad == 'Alta':
                return "Problemas Críticos de Contratos"
            elif 'masivo' in nombre_actual.lower():
                return "Gestión Masiva de Contratos"
            else:
                return "Modificaciones de Contratos"
        else:
            return "Gestión de Contratos"
    
    if 'lentitud' in texto_completo or 'lento' in texto_completo or 'rendimiento' in texto_completo:
        return "Problemas de Rendimiento del Sistema"
    
    if 'infraestructura' in texto_completo:
        if criticidad == 'Alta':
            return "Problemas Críticos de Infraestructura"
        else:
            return "Solicitudes de Infraestructura"
    
    if 'cups' in texto_completo and ('regularizar' in texto_completo or 'error' in texto_completo):
        return "Errores en Gestión de CUPS"
    
    if 'consulta' in texto_completo or 'prueba' in texto_completo:
        return "Consultas y Pruebas del Sistema"
    
    if 'puesto' in texto_completo and 'trabajo' in texto_completo:
        return "Solicitudes de Puestos de Trabajo"
    
    if 'apartadas' in texto_completo and 'regularizadoras' in texto_completo:
        return "Solicitudes de Datos Regulatorios"
    
    if 'renovaciones' in texto_completo or 'aparato' in texto_completo:
        return "Gestión de Equipos y Renovaciones"
    
    # Fallback: intentar generar nombre basado en conceptos
    if conceptos:
        # Identificar tipo de problema principal
        tipos_problema = ['errores', 'fallos', 'problemas', 'problemas críticos']
        tipo_encontrado = next((tipo for tipo in tipos_problema if tipo in conceptos), 'gestión')
        
        # Identificar área funcional principal
        areas_funcionales = ['facturación', 'contratos', 'sistema', 'datos', 'equipos', 'usuarios']
        area_encontrada = next((area for area in areas_funcionales if area in conceptos), 'sistema')
        
        if tipo_encontrado == 'gestión':
            return f"Gestión de {area_encontrada.title()}"
        else:
            return f"{tipo_encontrado.title()} en {area_encontrada.title()}"
    
    # Si no se puede mejorar, limpiar el nombre actual
    nombre_limpio = nombre_actual.replace('_', ' ')
    nombre_limpio = re.sub(r'\b(frecuentes|masivo|automatico)\b', '', nombre_limpio, flags=re.IGNORECASE)
    nombre_limpio = re.sub(r'\s+', ' ', nombre_limpio).strip()
    return nombre_limpio

def main():
    """Función principal"""
    print("🎯 MEJORANDO NOMBRES DE CATEGORÍAS DEL CLUSTERING")
    print("=" * 60)
    
    # Cargar datos del clustering
    with open('outputs_completo/data/analisis_completo_naturgy.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tipos_incidencia = data['tipos_de_incidencia']
    
    # Generar nuevos nombres
    nombres_mejorados = {}
    
    print("📋 TRANSFORMACIONES APLICADAS:")
    print()
    
    for key, categoria_info in tipos_incidencia.items():
        nombre_actual = categoria_info['nombre']
        nombre_mejorado = generar_nombre_contextual(categoria_info)
        
        nombres_mejorados[nombre_actual] = nombre_mejorado
        
        print(f"• {nombre_actual}")
        print(f"  → {nombre_mejorado}")
        print(f"    📊 {categoria_info['num_incidencias']} casos")
        print()
    
    # Guardar mapeo de nombres
    with open('mapeo_nombres_mejorados.json', 'w', encoding='utf-8') as f:
        json.dump(nombres_mejorados, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Mapeo guardado en: mapeo_nombres_mejorados.json")
    print(f"📊 Total de categorías mejoradas: {len(nombres_mejorados)}")
    
    return nombres_mejorados

if __name__ == "__main__":
    main()
