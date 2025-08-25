# Notebook de Clasificación de Aeronaves - Multiplataforma (Mac M4 + NVIDIA A100)

Crear un **Jupyter notebook completo** para clasificación de imágenes de aeronaves usando PyTorch + timm. El notebook debe funcionar en **Mac M4 (MPS) y NVIDIA A100 (CUDA)** con detección automática de plataforma y optimización.

**IMPORTANTE: Todo el notebook debe estar en ESPAÑOL** - comentarios, documentación, nombres de variables, texto explicativo, y salidas de print.

## **Especificaciones del Dataset**

- **Ubicación del dataset**: directorio `./aviones/`
- **Clases**: 100 clases de aeronaves (estructura ImageFolder)
- **División de datos**: **80% entrenamiento / 10% validación / 10% prueba**
- **División automática**: Crear splits train/val/test desde la carpeta `aviones` programáticamente

## **Estructura del Notebook (4 Secciones Principales)**

### **1. Configuración del Entorno y Verificación de GPU**
```python
# Detección de plataforma y optimización
# Verificación de GPU (MPS/CUDA/CPU)
# Verificación de instalación de dependencias
# Información de memoria y dispositivo
# Configurar semillas aleatorias para reproducibilidad
# Todo en español: print("GPU detectada:", dispositivo)
```

### **2. Definición de Dataset y Transformaciones**  
```python
# Auto-crear división 80/10/10 train/val/test desde ./aviones/
# Configuración de dataset ImageFolder
# Transformaciones adaptativas por plataforma:
#   - Entrenamiento: RandomResizedCrop(224), RandomHorizontalFlip, RandAugment, ColorJitter
#   - Validación/Prueba: Resize(256), CenterCrop(224)
# DataLoaders optimizados por plataforma con batch sizes y workers apropiados
# Visualización del dataset y análisis de distribución de clases
# Variables y comentarios en español: cargador_datos_entrenamiento
```

### **3. Baseline: ResNet-18 (Precisión Esperada: 55%)**
```python
# Cargar ResNet-18 pre-entrenado desde timm
# Congelar todas las capas (solo extracción de características)
# Entrenar por 10 épocas
# Bucle de entrenamiento adaptativo por plataforma con:
#   - Detección automática de dispositivo y optimización
#   - Loss de entropía cruzada + optimizador Adam
#   - Scheduler coseno con warmup
#   - Seguimiento de progreso y visualización
#   - 📊 IMPORTANTE: Guardar métricas en historial_entrenamiento['baseline']
# Evaluación en conjunto de validación
# Precisión esperada: ~55%
# Todo documentado en español: epoca, precision, perdida

# En cada época del baseline:
guardar_metricas_epoca(historial_entrenamiento, 'baseline', epoca, 
                      loss_entrenamiento, loss_validacion, 
                      precision_entrenamiento, precision_validacion)
```

### **4. ResNet-18 con Unfreezing (Precisión Esperada: 64%)**
```python  
# Descongelar capas específicas: ['layer4', 'fc']
# Tasas de aprendizaje discriminativas:
#   - Capas congeladas: 0 (sin entrenamiento)
#   - layer4: 1e-4  
#   - fc (clasificador): 3e-4
# Entrenar por 10 épocas
# Entrenamiento mejorado con:
#   - Optimización específica por capa
#   - Monitoreo avanzado
#   - Programación de tasa de aprendizaje por grupo de capas
#   - 📊 IMPORTANTE: Guardar métricas en historial_entrenamiento['descongelado']
# Evaluación final en conjunto de prueba
# Precisión esperada: ~64%
# Variables en español: capas_descongeladas, optimizador_discriminativo

# En cada época del modelo descongelado:
guardar_metricas_epoca(historial_entrenamiento, 'descongelado', epoca, 
                      loss_entrenamiento, loss_validacion, 
                      precision_entrenamiento, precision_validacion)

# Al final del entrenamiento:
crear_graficas_comparativas(historial_entrenamiento)
crear_grafica_loss_combinada(historial_entrenamiento)
exportar_historial_csv(historial_entrenamiento, 'metricas_aeronaves.csv')
```

## **Optimizaciones Multiplataforma**

### **Configuración de Dispositivo**
```python
def obtener_configuracion_plataforma():
    if torch.cuda.is_available():
        # Configuración NVIDIA A100
        return {
            'dispositivo': 'cuda',
            'tamano_lote': 128,
            'num_trabajadores': 16,
            'pin_memory': True,
            'tasa_aprendizaje': 3e-4
        }
    elif torch.backends.mps.is_available():
        # Configuración Mac M4  
        return {
            'dispositivo': 'mps',
            'tamano_lote': 32,
            'num_trabajadores': 4,
            'pin_memory': False,
            'tasa_aprendizaje': 2e-4
        }
    else:
        # Respaldo CPU
        return {
            'dispositivo': 'cpu',
            'tamano_lote': 16,
            'num_trabajadores': 2,
            'pin_memory': False,
            'tasa_aprendizaje': 1e-4
        }
```

## **Requisitos Técnicos**

### **Gestión de Datos**
- **Creación automática de división**: Usar `train_test_split` con estratificación
- **Estructura de directorios**: 
  ```
  aviones/
  ├── clase_001/
  ├── clase_002/
  └── ...
  ```
- **Preservación de división**: Guardar índices de división para experimentos reproducibles

### **Características de Entrenamiento**
- **Precisión mixta**: Implementación de AMP apropiada por plataforma
- **Gestión de memoria**: Limpieza regular de caché por plataforma
- **Seguimiento de progreso**: Visualizaciones ricas con matplotlib/seaborn
- **Registro de métricas**: Curvas de precisión y pérdida, matrices de confusión
- **Checkpointing**: Guardar mejores modelos para cada experimento
- **📊 Guardado de datos para gráficas**: Almacenar historial completo de entrenamiento para visualizaciones posteriores

### **Almacenamiento de Datos de Entrenamiento**
```python
# Estructura para guardar métricas de ambos modelos
historial_entrenamiento = {
    'baseline': {
        'loss_entrenamiento': [],
        'loss_validacion': [],
        'precision_entrenamiento': [],
        'precision_validacion': [],
        'epocas': []
    },
    'descongelado': {
        'loss_entrenamiento': [],
        'loss_validacion': [],
        'precision_entrenamiento': [],
        'precision_validacion': [],
        'epocas': []
    }
}

# Guardar datos en cada época
def guardar_metricas_epoca(historial, modelo_tipo, epoca, loss_train, loss_val, acc_train, acc_val):
    historial[modelo_tipo]['epocas'].append(epoca)
    historial[modelo_tipo]['loss_entrenamiento'].append(loss_train)
    historial[modelo_tipo]['loss_validacion'].append(loss_val)
    historial[modelo_tipo]['precision_entrenamiento'].append(acc_train)
    historial[modelo_tipo]['precision_validacion'].append(acc_val)

# Exportar datos a CSV para análisis posterior
import pandas as pd
def exportar_historial_csv(historial, nombre_archivo='historial_entrenamiento.csv'):
    datos_completos = []
    for modelo in historial.keys():
        for i in range(len(historial[modelo]['epocas'])):
            datos_completos.append({
                'modelo': modelo,
                'epoca': historial[modelo]['epocas'][i],
                'loss_entrenamiento': historial[modelo]['loss_entrenamiento'][i],
                'loss_validacion': historial[modelo]['loss_validacion'][i],
                'precision_entrenamiento': historial[modelo]['precision_entrenamiento'][i],
                'precision_validacion': historial[modelo]['precision_validacion'][i]
            })
    
    df = pd.DataFrame(datos_completos)
    df.to_csv(nombre_archivo, index=False)
    print(f"Historial guardado en: {nombre_archivo}")
```

### **Visualización y Análisis**
- **Exploración del dataset**: Distribución de clases, imágenes de muestra
- **Monitoreo de entrenamiento**: Gráficos de pérdida/precisión en tiempo real
- **Comparación de modelos**: Resultados lado a lado baseline vs descongelado
- **Análisis de errores**: Muestras mal clasificadas, rendimiento por clase
- **Resultados finales**: Evaluación comprensiva del conjunto de prueba
- **📈 Gráficas comparativas finales**: Curvas Loss vs Epoch para ambos modelos en una sola visualización

### **Funciones de Visualización Comparativa**
```python
def crear_graficas_comparativas(historial):
    """
    Crear gráficas comparativas de Loss vs Epoch para ambos modelos
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfica 1: Loss de Entrenamiento Comparativo
    ax1.plot(historial['baseline']['epocas'], historial['baseline']['loss_entrenamiento'], 
             'b-', label='Baseline ResNet-18', linewidth=2)
    ax1.plot(historial['descongelado']['epocas'], historial['descongelado']['loss_entrenamiento'], 
             'r-', label='ResNet-18 Descongelado', linewidth=2)
    ax1.set_title('Pérdida de Entrenamiento vs Época', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Loss de Validación Comparativo
    ax2.plot(historial['baseline']['epocas'], historial['baseline']['loss_validacion'], 
             'b--', label='Baseline ResNet-18', linewidth=2)
    ax2.plot(historial['descongelado']['epocas'], historial['descongelado']['loss_validacion'], 
             'r--', label='ResNet-18 Descongelado', linewidth=2)
    ax2.set_title('Pérdida de Validación vs Época', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Precisión de Entrenamiento Comparativo
    ax3.plot(historial['baseline']['epocas'], historial['baseline']['precision_entrenamiento'], 
             'b-', label='Baseline ResNet-18', linewidth=2)
    ax3.plot(historial['descongelado']['epocas'], historial['descongelado']['precision_entrenamiento'], 
             'r-', label='ResNet-18 Descongelado', linewidth=2)
    ax3.set_title('Precisión de Entrenamiento vs Época', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Precisión (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Precisión de Validación Comparativo
    ax4.plot(historial['baseline']['epocas'], historial['baseline']['precision_validacion'], 
             'b--', label='Baseline ResNet-18', linewidth=2)
    ax4.plot(historial['descongelado']['epocas'], historial['descongelado']['precision_validacion'], 
             'r--', label='ResNet-18 Descongelado', linewidth=2)
    ax4.set_title('Precisión de Validación vs Época', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Precisión (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráficas comparativas guardadas como 'comparacion_modelos.png'")

def crear_grafica_loss_combinada(historial):
    """
    Crear una sola gráfica con todas las curvas de pérdida
    """
    plt.figure(figsize=(12, 8))
    
    # Curvas de pérdida
    plt.plot(historial['baseline']['epocas'], historial['baseline']['loss_entrenamiento'], 
             'b-', label='Baseline - Entrenamiento', linewidth=2)
    plt.plot(historial['baseline']['epocas'], historial['baseline']['loss_validacion'], 
             'b--', label='Baseline - Validación', linewidth=2)
    plt.plot(historial['descongelado']['epocas'], historial['descongelado']['loss_entrenamiento'], 
             'r-', label='Descongelado - Entrenamiento', linewidth=2)
    plt.plot(historial['descongelado']['epocas'], historial['descongelado']['loss_validacion'], 
             'r--', label='Descongelado - Validación', linewidth=2)
    
    plt.title('Comparación de Pérdida: Baseline vs Descongelado', fontsize=16, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Añadir anotaciones de mejora
    min_loss_baseline = min(historial['baseline']['loss_validacion'])
    min_loss_descongelado = min(historial['descongelado']['loss_validacion'])
    mejora = ((min_loss_baseline - min_loss_descongelado) / min_loss_baseline) * 100
    
    plt.text(0.6, 0.8, f'Mejora en pérdida: {mejora:.1f}%', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('loss_comparativo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráfica de pérdida comparativa guardada como 'loss_comparativo.png'")
```

## **Estructura de Celdas del Notebook**

```python
# Celda 1: Configuración del entorno e importaciones
import torch, torchvision, timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
# Detección de plataforma y configuración
# Inicialización de historial_entrenamiento
# Todo en español: print("Configurando entorno...")

# Celda 2: Preparación de datos  
# Auto-crear divisiones train/val/test
# Creación de Dataset y DataLoader
# Variables en español: conjunto_entrenamiento, cargador_datos

# Celdas 3-N: Entrenamiento ResNet-18 baseline (múltiples celdas)
# Creación del modelo, bucle de entrenamiento, evaluación
# 📊 Guardado de métricas en cada época: guardar_metricas_epoca()
# Comentarios en español: "Iniciando entrenamiento baseline..."

# Celdas N+1-M: Experimento de descongelamiento (múltiples celdas)  
# Descongelamiento de capas, LRs discriminativos, entrenamiento
# 📊 Guardado de métricas en cada época para modelo descongelado
# Variables: modelo_descongelado, capas_entrenables

# Celdas finales: Comparación y análisis
# 📈 Llamada a crear_graficas_comparativas(historial_entrenamiento)
# 📈 Llamada a crear_grafica_loss_combinada(historial_entrenamiento)
# 📄 Exportación a CSV: exportar_historial_csv()
# 💾 Guardado de historial completo en pickle
# Todo documentado en español

# Celda final: Resumen de archivos generados
print("🎯 Análisis completado!")
print("📊 Gráficas guardadas: comparacion_modelos.png, loss_comparativo.png")
print("📄 Datos exportados: metricas_aeronaves.csv")
print("💾 Modelos guardados: modelo_baseline.pth, modelo_descongelado.pth")
```

## **Salidas Esperadas**

### **Baseline (Sección 3)**
- Curvas de entrenamiento mostrando convergencia
- Precisión de validación: ~55%
- Matriz de confusión y métricas por clase
- Checkpoint del modelo guardado
- 📊 **Datos guardados**: Historial completo en estructura `historial_entrenamiento['baseline']`
- Mensajes en español: "Epoch 1/10 - Pérdida: 2.45, Precisión: 23.4%"

### **Modelo Descongelado (Sección 4)**  
- Visualización de tasas de aprendizaje por capa
- Curvas de entrenamiento con rendimiento mejorado
- Precisión de prueba: ~64% 
- Análisis detallado de errores
- Resumen de comparación de modelos
- 📊 **Datos guardados**: Historial completo en estructura `historial_entrenamiento['descongelado']`
- 📈 **Gráficas comparativas finales**: 
  - Gráfica de 4 paneles comparando ambos modelos
  - Gráfica combinada de pérdida vs época
  - Archivo CSV con todos los datos: `metricas_aeronaves.csv`
- Todo en español: "Mejora de precisión: +9% respecto al baseline"

### **Archivos Generados**
```
📁 Salidas del notebook:
├── 📊 comparacion_modelos.png (4 gráficas en una imagen)
├── 📈 loss_comparativo.png (gráfica combinada de pérdida)  
├── 📄 metricas_aeronaves.csv (datos para análisis posterior)
├── 💾 modelo_baseline.pth (checkpoint baseline)
├── 💾 modelo_descongelado.pth (checkpoint descongelado)
└── 🔧 historial_entrenamiento.pkl (objeto Python completo)
```

## **Installation Requirements**

### **For Mac M4**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm matplotlib seaborn scikit-learn pandas numpy jupyter
```

### **For NVIDIA A100**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  
pip install timm matplotlib seaborn scikit-learn pandas numpy jupyter
```

## **Características Principales**

1. **📊 Visualizaciones Ricas**: Gráficos interactivos, barras de progreso, matrices de confusión
2. **🔄 Detección Automática de Plataforma**: Funciona sin problemas en Mac M4 o A100
3. **📈 Análisis Comprensivo**: Comparación detallada de rendimiento y análisis de errores
4. **🎯 Resultados Reproducibles**: Semillas apropiadas y preservación de divisiones
5. **💾 Checkpointing Inteligente**: Guardar y cargar mejores modelos automáticamente
6. **📝 Documentación Clara**: Celdas bien comentadas con explicaciones en español

**Objetivo**: Notebook completo de Jupyter para clasificación de aeronaves que optimiza automáticamente para el hardware disponible mientras proporciona análisis y visualización comprensivos del proceso de entrenamiento. **Todo en idioma español.**