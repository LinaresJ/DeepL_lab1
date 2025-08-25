#!/usr/bin/env python3
"""
Script para entrenar los 3 modelos ResNet (ResNet-18, ResNet-34, ResNet-50)
usando las mismas configuraciones del notebook 1_clasificacion_baseline.ipynb

Autor: Generado desde notebook de clasificación de aeronaves
Dataset: 100 clases de aeronaves con división 80/10/10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from tqdm.auto import tqdm
import time
import os

# Configuración de estilo y warnings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

def configurar_dispositivo():
    """
    Detecta y configura automáticamente el mejor dispositivo disponible.
    Prioridad: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 Usando CUDA: {gpu_name}")
        print(f"💾 Memoria GPU: {memory_gb:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Usando Apple Silicon MPS")
        print("💾 Memoria compartida con sistema")
    else:
        device = torch.device('cpu')
        print("💻 Usando CPU (advertencia: entrenamiento será lento)")
    
    # Configuraciones adicionales para rendimiento
    torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False
    
    return device

def crear_transformaciones_adaptativas(input_size):
    """
    Crea transformaciones adaptadas al tamaño de entrada específico
    """
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def crear_dataloaders_adaptativos(batch_size, input_size, dataset_path, num_workers):
    """
    Crea dataloaders con configuraciones específicas de batch size e input size
    """
    train_transforms_adapt, val_transforms_adapt = crear_transformaciones_adaptativas(input_size)
    
    # Datasets con transformaciones adaptativas
    train_dataset_adapt = ImageFolder(
        root=dataset_path / 'train',
        transform=train_transforms_adapt
    )
    
    val_dataset_adapt = ImageFolder(
        root=dataset_path / 'val',
        transform=val_transforms_adapt
    )
    
    # DataLoaders con batch_size específico
    train_loader_adapt = DataLoader(
        train_dataset_adapt,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() or hasattr(torch.backends, 'mps') else False
    )
    
    val_loader_adapt = DataLoader(
        val_dataset_adapt,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() or hasattr(torch.backends, 'mps') else False
    )
    
    return train_loader_adapt, val_loader_adapt, train_dataset_adapt

def crear_modelo_evaluacion(modelo_timm, num_classes):
    """
    Crea modelo con fine-tuning parcial para evaluación rápida
    """
    # Crear modelo preentrenado
    modelo = timm.create_model(modelo_timm, pretrained=True)
    
    # Congelar todas las capas inicialmente
    for param in modelo.parameters():
        param.requires_grad = False
    
    # Para ResNet: descongelar layer4 y fc
    if hasattr(modelo, 'layer4'):
        for param in modelo.layer4.parameters():
            param.requires_grad = True
    
    # Reemplazar clasificador final
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    for param in modelo.fc.parameters():
        param.requires_grad = True
    
    return modelo

def contar_parametros(model):
    """
    Cuenta parámetros entrenables y totales del modelo
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def entrenar_modelo_evaluacion(modelo, train_loader, val_loader, num_epochs, learning_rate, nombre_modelo, device):
    """
    Versión optimizada de entrenamiento para evaluación rápida
    """
    # Configurar optimizador y criterio
    optimizador = optim.AdamW(modelo.parameters(), lr=learning_rate, weight_decay=0.01)
    criterio = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizador, max_lr=learning_rate, 
        epochs=num_epochs, steps_per_epoch=len(train_loader)
    )
    
    # Historial de métricas
    historial = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    mejor_val_acc = 0.0
    
    print(f"🚀 Entrenando {nombre_modelo} ({num_epochs} épocas)")
    print("-" * 50)
    
    for epoca in range(num_epochs):
        tiempo_inicio = time.time()
        
        # =======================================
        # ENTRENAMIENTO
        # =======================================
        modelo.train()
        train_loss_total = 0.0
        train_correctos = 0
        train_total = 0
        
        barra_train = tqdm(train_loader, desc=f'Época {epoca+1}/{num_epochs} [Train]', leave=False)
        for inputs, targets in barra_train:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizador.zero_grad()
            outputs = modelo(inputs)
            loss = criterio(outputs, targets)
            loss.backward()
            optimizador.step()
            scheduler.step()
            
            train_loss_total += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correctos += predicted.eq(targets).sum().item()
            
            acc_actual = 100. * train_correctos / train_total
            barra_train.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{acc_actual:.1f}%'})
        
        train_loss_prom = train_loss_total / len(train_loader)
        train_accuracy = 100. * train_correctos / train_total
        
        # =======================================
        # VALIDACIÓN  
        # =======================================
        modelo.eval()
        val_loss_total = 0.0
        val_correctos = 0
        val_total = 0
        
        with torch.no_grad():
            barra_val = tqdm(val_loader, desc=f'Época {epoca+1}/{num_epochs} [Val]', leave=False)
            for inputs, targets in barra_val:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = modelo(inputs)
                loss = criterio(outputs, targets)
                
                val_loss_total += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correctos += predicted.eq(targets).sum().item()
                
                acc_actual = 100. * val_correctos / val_total
                barra_val.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{acc_actual:.1f}%'})
        
        val_loss_prom = val_loss_total / len(val_loader)
        val_accuracy = 100. * val_correctos / val_total
        
        # Guardar métricas
        historial['train_loss'].append(train_loss_prom)
        historial['train_acc'].append(train_accuracy)
        historial['val_loss'].append(val_loss_prom)
        historial['val_acc'].append(val_accuracy)
        
        # Actualizar mejor accuracy
        if val_accuracy > mejor_val_acc:
            mejor_val_acc = val_accuracy
            mejora_icon = "📈"
        else:
            mejora_icon = "📊"
        
        tiempo_epoca = time.time() - tiempo_inicio
        
        print(f"Época {epoca+1:2d}/{num_epochs} {mejora_icon} | "
              f"Train: Loss={train_loss_prom:.4f}, Acc={train_accuracy:5.1f}% | "
              f"Val: Loss={val_loss_prom:.4f}, Acc={val_accuracy:5.1f}% | "
              f"Tiempo: {tiempo_epoca:.1f}s")
    
    print(f"✅ {nombre_modelo} completado. Mejor Val Acc: {mejor_val_acc:.1f}%")
    print()
    
    return modelo, historial, mejor_val_acc

def crear_tabla_comparativa(resultados, num_epocas_eval):
    """
    Crea tabla comparativa con todos los resultados
    """
    datos_tabla = []
    
    for modelo_key, resultado in resultados.items():
        if 'error' not in resultado:
            datos_tabla.append({
                'Modelo': resultado['nombre'],
                'Precisión Val (%)': f"{resultado['mejor_val_acc']:.1f}",
                'Parámetros Total': f"{resultado['total_params']:,}",
                'Parámetros Entrenables': f"{resultado['trainable_params']:,}",
                'Batch Size': resultado['config']['batch_size'],
                'Input Size': f"{resultado['config']['input_size']}px",
                'Épocas': num_epocas_eval
            })
        else:
            datos_tabla.append({
                'Modelo': resultado['nombre'],
                'Precisión Val (%)': "ERROR",
                'Parámetros Total': "-",
                'Parámetros Entrenables': "-",
                'Batch Size': "-",
                'Input Size': "-",
                'Épocas': num_epocas_eval
            })
    
    # Ordenar por precisión (descendente)
    datos_tabla.sort(key=lambda x: float(x['Precisión Val (%)'].replace('%', '')) if x['Precisión Val (%)'] != "ERROR" else 0, reverse=True)
    
    df_comparativa = pd.DataFrame(datos_tabla)
    return df_comparativa

def crear_grafica_comparativa_resnet(resultados):
    """
    Crea gráfica de barras específica para modelos ResNet
    """
    # Extraer datos para gráfica
    nombres_modelos = []
    precisiones = []
    
    for modelo_key, resultado in resultados.items():
        if 'error' not in resultado:
            nombres_modelos.append(resultado['nombre'])
            precisiones.append(resultado['mejor_val_acc'])
    
    # Ordenar por precisión
    datos_ordenados = sorted(zip(nombres_modelos, precisiones), key=lambda x: x[1], reverse=True)
    nombres_ord, precisiones_ord = zip(*datos_ordenados)
    
    # Crear gráfica
    plt.figure(figsize=(12, 8))
    
    # Colores específicos para ResNet
    colores_resnet = ['#2E86AB', '#1F5582', '#3B9AC9']
    barras = plt.bar(range(len(nombres_ord)), precisiones_ord, 
                    color=colores_resnet[:len(nombres_ord)], alpha=0.8, 
                    edgecolor='black', linewidth=1)
    
    # Personalizar gráfica
    plt.title('Comparación de Rendimiento: Modelos ResNet - Clasificación de Aeronaves', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Modelos ResNet', fontsize=12, fontweight='bold')
    plt.ylabel('Precisión de Validación (%)', fontsize=12, fontweight='bold')
    
    # Configurar eje X
    plt.xticks(range(len(nombres_ord)), nombres_ord, rotation=0)
    
    # Añadir valores sobre las barras
    for i, (barra, precision) in enumerate(zip(barras, precisiones_ord)):
        plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.5, 
                f'{precision:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Añadir línea de referencia (promedio)
    if len(precisiones_ord) > 1:
        promedio = np.mean(precisiones_ord)
        plt.axhline(y=promedio, color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(len(nombres_ord)-1, promedio + 1, f'Promedio: {promedio:.1f}%', 
                 ha='right', va='bottom', color='red', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Guardar gráfica
    plt.savefig('comparacion_resnet_modelos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Gráfica guardada como: comparacion_resnet_modelos.png")

def main():
    """
    Función principal para ejecutar el entrenamiento de los 3 modelos ResNet
    """
    print("🎯 ENTRENAMIENTO DE MODELOS RESNET PARA CLASIFICACIÓN DE AERONAVES")
    print("="*70)
    
    # Configurar dispositivo
    device = configurar_dispositivo()
    print(f"✅ Dispositivo configurado: {device}\n")
    
    # Configurar dataset path
    dataset_path = Path('./data')
    
    # Verificar estructura del dataset
    print("📁 Verificando estructura del dataset...")
    if not dataset_path.exists():
        print(f"❌ Error: No se encontró el directorio {dataset_path.absolute()}")
        print("   Asegúrate de que el dataset esté en './data/' con subdirectorios 'train', 'val', 'test'")
        return
    
    # Obtener número de clases
    train_path = dataset_path / 'train'
    if train_path.exists():
        num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
        print(f"✅ Dataset encontrado: {num_classes} clases")
    else:
        print("❌ Error: No se encontró el directorio 'train'")
        return
    
    # Configuraciones específicas para cada modelo ResNet
    configuraciones_modelos = {
        'resnet18': {
            'batch_size': 32,
            'input_size': 224,
            'modelo_timm': 'resnet18',
            'nombre_display': 'ResNet-18'
        },
        'resnet34': {
            'batch_size': 32,
            'input_size': 224,
            'modelo_timm': 'resnet34',
            'nombre_display': 'ResNet-34'
        },
        'resnet50': {
            'batch_size': 24,
            'input_size': 224,
            'modelo_timm': 'resnet50',
            'nombre_display': 'ResNet-50'
        }
    }
    
    # Parámetros de evaluación (mismos del notebook)
    NUM_EPOCAS_EVAL = 3
    LEARNING_RATE_EVAL = 0.001
    NUM_WORKERS = 4
    
    print(f"\n⚙️ Configuración de entrenamiento:")
    print(f"  Épocas por modelo: {NUM_EPOCAS_EVAL}")
    print(f"  Tasa de aprendizaje: {LEARNING_RATE_EVAL}")
    print(f"  Trabajadores: {NUM_WORKERS}")
    
    print(f"\n⚙️ Configuraciones por modelo:")
    for modelo, config in configuraciones_modelos.items():
        print(f"  {config['nombre_display']}: Batch={config['batch_size']}, Input={config['input_size']}px")
    
    # Ejecutar evaluación de los 3 modelos ResNet
    print(f"\n🎯 INICIANDO ENTRENAMIENTO DE MODELOS RESNET")
    print("="*60)
    
    resultados_evaluacion = {}
    modelos_entrenados = {}
    
    for modelo_key, config in configuraciones_modelos.items():
        print(f"\n🤖 Entrenando {config['nombre_display']}...")
        print(f"   Configuración: Batch={config['batch_size']}, Input={config['input_size']}px")
        
        try:
            # Crear dataloaders específicos para este modelo
            train_loader_eval, val_loader_eval, train_dataset_eval = crear_dataloaders_adaptativos(
                config['batch_size'], config['input_size'], dataset_path, NUM_WORKERS
            )
            
            # Crear modelo
            modelo_eval = crear_modelo_evaluacion(config['modelo_timm'], num_classes)
            modelo_eval = modelo_eval.to(device)
            
            # Contar parámetros
            total_params, trainable_params = contar_parametros(modelo_eval)
            print(f"   Parámetros entrenables: {trainable_params:,} de {total_params:,} ({100*trainable_params/total_params:.1f}%)")
            
            # Entrenar modelo
            modelo_entrenado, historial_eval, mejor_acc = entrenar_modelo_evaluacion(
                modelo_eval, train_loader_eval, val_loader_eval, 
                NUM_EPOCAS_EVAL, LEARNING_RATE_EVAL, config['nombre_display'], device
            )
            
            # Guardar resultados
            resultados_evaluacion[modelo_key] = {
                'nombre': config['nombre_display'],
                'mejor_val_acc': mejor_acc,
                'historial': historial_eval,
                'config': config,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            modelos_entrenados[modelo_key] = modelo_entrenado
            
        except Exception as e:
            print(f"❌ Error entrenando {config['nombre_display']}: {str(e)}")
            resultados_evaluacion[modelo_key] = {
                'nombre': config['nombre_display'],
                'mejor_val_acc': 0.0,
                'error': str(e)
            }
    
    print("\n🎉 ENTRENAMIENTO COMPLETADO")
    print("="*60)
    
    # Crear y mostrar tabla comparativa
    print(f"\n📊 TABLA COMPARATIVA DE RESULTADOS - MODELOS RESNET ({NUM_EPOCAS_EVAL} ÉPOCAS)")
    print("="*80)
    
    df_resultados = crear_tabla_comparativa(resultados_evaluacion, NUM_EPOCAS_EVAL)
    print(df_resultados.to_string(index=False))
    print("="*80)
    
    # Guardar tabla en CSV
    df_resultados.to_csv('evaluacion_resnet_modelos.csv', index=False)
    print(f"\n📄 Tabla guardada en: evaluacion_resnet_modelos.csv")
    
    # Estadísticas generales
    precisiones_exitosas = [r['mejor_val_acc'] for r in resultados_evaluacion.values() if 'error' not in r]
    
    if precisiones_exitosas:
        mejor_modelo = max(resultados_evaluacion.items(), key=lambda x: x[1]['mejor_val_acc'] if 'error' not in x[1] else 0)
        peor_modelo = min(resultados_evaluacion.items(), key=lambda x: x[1]['mejor_val_acc'] if 'error' not in x[1] else float('inf'))
        
        print(f"\n🔍 ANÁLISIS DE RESULTADOS:")
        print(f"🏆 MEJOR MODELO: {mejor_modelo[1]['nombre']}: {mejor_modelo[1]['mejor_val_acc']:.1f}%")
        print(f"📉 MENOR RENDIMIENTO: {peor_modelo[1]['nombre']}: {peor_modelo[1]['mejor_val_acc']:.1f}%")
        print(f"📊 PRECISIÓN PROMEDIO: {np.mean(precisiones_exitosas):.1f}%")
        print(f"📈 RANGO: {np.max(precisiones_exitosas):.1f}% - {np.min(precisiones_exitosas):.1f}%")
    
    # Crear gráfica comparativa
    print(f"\n📊 Generando gráfica comparativa...")
    crear_grafica_comparativa_resnet(resultados_evaluacion)
    
    # Exportar historial detallado
    datos_detallados = []
    for modelo_key, resultado in resultados_evaluacion.items():
        if 'error' not in resultado:
            for epoca in range(NUM_EPOCAS_EVAL):
                datos_detallados.append({
                    'Modelo': resultado['nombre'],
                    'Epoca': epoca + 1,
                    'Train_Loss': resultado['historial']['train_loss'][epoca],
                    'Train_Acc': resultado['historial']['train_acc'][epoca],
                    'Val_Loss': resultado['historial']['val_loss'][epoca], 
                    'Val_Acc': resultado['historial']['val_acc'][epoca]
                })
    
    if datos_detallados:
        df_detallado = pd.DataFrame(datos_detallados)
        df_detallado.to_csv('historial_resnet_modelos_detallado.csv', index=False)
        
        print(f"\n📄 Archivos generados:")
        print(f"  - evaluacion_resnet_modelos.csv (resumen)")
        print(f"  - historial_resnet_modelos_detallado.csv (épocas detalladas)")
        print(f"  - comparacion_resnet_modelos.png (gráfica)")
    
    print(f"\n✅ Entrenamiento de modelos ResNet completado exitosamente!")
    print("="*70)

if __name__ == "__main__":
    main()