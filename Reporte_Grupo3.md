# Reporte de Resultados Preliminares - Clasificación de Aeronaves
## Grupo 3

### 📋 Resumen Ejecutivo

Este reporte presenta los resultados preliminares de la evaluación de diferentes arquitecturas de redes neuronales convolucionales para la clasificación de imágenes de aeronaves. Se evaluaron 8 modelos diferentes utilizando un dataset con 100 clases de aeronaves, con 8,000 imágenes de entrenamiento y 1,000 de validación.

### 🎯 Objetivos

- Comparar el rendimiento de diferentes arquitecturas de CNN modernas
- Identificar el modelo con mejor balance entre precisión y eficiencia
- Evaluar la convergencia de los modelos con entrenamiento extendido (30 épocas intentadas)
- Implementar entrenamiento con técnicas avanzadas de optimización

### 📊 Resultados Obtenidos

#### Tabla Comparativa de Rendimiento

| Modelo | Épocas | Top-1 Accuracy (Val) | Top-5 Accuracy (Val) | Estado | Observaciones |
|--------|---------|---------------------|---------------------|---------|---------------|
| **EfficientNet-B1** | 3 | **70.70%** | **92.40%** | ✅ Exitoso | Mejor rendimiento general (entrenamiento inicial) |
| **EfficientNet-B2** | 3 | **68.00%** | **90.50%** | ✅ Exitoso | Segundo mejor (entrenamiento inicial) |
| **DenseNet-121** | 3 | **61.70%** | **86.80%** | ✅ Exitoso | Buen rendimiento (entrenamiento inicial) |
| **EfficientNet-B0** | 3 | **51.60%** | **83.50%** | ✅ Exitoso | Modelo más liviano |
| **EfficientNet-B1** | 30 | - | - | ❌ Falla técnica | Error en función de pérdida durante validación |
| **EfficientNet-B2** | 30 | - | - | ❌ Falla técnica | Error en función de pérdida durante validación |
| **DenseNet-121** | 30 | - | - | ❌ Falla técnica | Error en función de pérdida durante validación |
| **ResNet-34 Plus** | 3 | 1.30% | 5.40% | ✅ Convergencia lenta | Necesita más épocas |
| **ResNet-18 Plus** | 3 | 0.70% | 4.50% | ✅ Convergencia lenta | Necesita más épocas |

#### Configuraciones de Entrenamiento

| Modelo | Batch Size | Input Size | Grad. Accumulation | EMA |
|--------|------------|------------|-------------------|-----|
| EfficientNet-B0 | 32 | 224×224 | 4 steps | No |
| EfficientNet-B1 | 24 | 240×240 | 5 steps | No |
| EfficientNet-B2 | 16 | 260×260 | 8 steps | No |
| DenseNet-121 | 24 | 224×224 | 5 steps | No |
| ResNet-18 Plus | 32 | 224×224 | 4 steps | Sí |
| ResNet-34 Plus | 24 | 224×224 | 5 steps | Sí |

### 🔧 Definiciones de Modelos

#### EfficientNet-B1
- **Arquitectura**: EfficientNet-B1 con escalado compuesto de profundidad, anchura y resolución
- **Parámetros**: ~7.8M parámetros
- **Entrada**: 240×240 píxeles
- **Características**: 
  - Arquitectura basada en Neural Architecture Search (NAS)
  - Uso de bloques MBConv con Squeeze-and-Excitation
  - Escalado eficiente de todos los componentes de la red

#### EfficientNet-B2
- **Arquitectura**: EfficientNet-B2 con mayor resolución y capacidad que B1
- **Parámetros**: ~9.2M parámetros
- **Entrada**: 260×260 píxeles
- **Características**:
  - Mayor profundidad y anchura que EfficientNet-B1
  - Mejor capacidad de representación a costa de más recursos computacionales
  - Activaciones Swish para mejor gradiente

#### DenseNet-121
- **Arquitectura**: DenseNet con 121 capas y conexiones densas
- **Parámetros**: ~8.0M parámetros
- **Entrada**: 224×224 píxeles
- **Características**:
  - Conexiones densas entre todas las capas
  - Reutilización eficiente de características
  - Reducción del problema de gradiente desvaneciente

### 📈 Análisis de Resultados

#### 🏆 Modelos Exitosos

1. **EfficientNet-B1** (Ganador)
   - **Precisión Top-1**: 70.70%
   - **Precisión Top-5**: 92.40%
   - Excelente convergencia en solo 3 épocas
   - Balance óptimo entre complejidad y rendimiento

2. **EfficientNet-B2**
   - **Precisión Top-1**: 68.00%
   - **Precisión Top-5**: 90.50%
   - Rendimiento muy competitivo
   - Requiere más recursos computacionales

3. **DenseNet-121**
   - **Precisión Top-1**: 61.70%
   - **Precisión Top-5**: 86.80%
   - Arquitectura robusta con buena generalización
   - Convergencia estable

#### ⚠️ Modelos con Dificultades

1. **Familia ResNet**
   - Los modelos ResNet mostraron convergencia muy lenta
   - Diferencia significativa entre precisión de validación y test
   - **ResNet-18 Base** falló por error dimensional
   - Requieren mayor número de épocas para convergencia adecuada

#### 🔧 Desafíos Técnicos Encontrados

1. **Entrenamiento Extendido (30 épocas)**
   - **Problema**: Error en la función de pérdida durante la fase de validación
   - **Error específico**: Incompatibilidad entre `SoftTargetCrossEntropy` (usado para Mixup) y targets regulares en validación
   - **Síntoma**: Falla después del primer epoch durante `validate()` en `loss = criterion(outputs, targets)`
   - **Impacto**: Imposibilitó completar el entrenamiento extendido para los tres modelos principales

2. **Configuración de Mixup/Augmentation**
   - El script usa `SoftTargetCrossEntropy` cuando Mixup está habilitado
   - En validación se esperan targets regulares (enteros) pero la función de pérdida espera distribuciones suaves
   - Requiere separación de criterios de pérdida para entrenamiento y validación

### 🔍 Observaciones Técnicas

#### Convergencia y Aprendizaje

- **EfficientNet**: Familia que mostró la mejor convergencia con rápida mejora desde la primera época
- **DenseNet**: Convergencia progresiva y estable a lo largo de las épocas
- **ResNet**: Convergencia extremadamente lenta, sugiriendo necesidad de:
  - Mayor número de épocas
  - Ajuste de hiperparámetros
  - Revisión de la implementación

#### Errores Identificados

1. **ResNet-18 Base**: Error dimensional en tensores (64 vs 100)
   - Posible problema en la capa de clasificación final
   - Requiere revisión de la arquitectura

### 📋 Configuración Experimental

- **Dataset**: 100 clases de aeronaves
- **Entrenamiento**: 8,000 imágenes
- **Validación**: 1,000 imágenes  
- **Épocas**: 3 (entrenamiento preliminar exitoso), 30 (intentado, fallas técnicas)
- **Hardware**: Apple Silicon GPU (MPS)
- **Optimización**: Gradient accumulation, Mixup, CutMix, EMA (para modelos Plus)
- **Técnicas Avanzadas**: SAM optimizer, Cosine Annealing LR, AutoAugment

### 🎯 Conclusiones y Recomendaciones

#### Conclusiones Principales

1. **EfficientNet-B1** demostró ser el modelo más efectivo para esta tarea
2. La familia **EfficientNet** superó consistentemente a otras arquitecturas
3. **DenseNet-121** ofrece un buen balance como alternativa
4. Los modelos **ResNet** requieren optimización adicional

#### Recomendaciones para Trabajo Futuro

1. **Resolución de Problemas Técnicos**
   - **Prioridad Alta**: Corregir incompatibilidad de función de pérdida en validación
   - Implementar criterios de pérdida separados para entrenamiento (SoftTargetCrossEntropy) y validación (CrossEntropyLoss)
   - Verificar manejo correcto de targets en modo validación

2. **Entrenamiento Extendido Exitoso**
   - Una vez corregidos los errores técnicos, completar entrenamiento de 30 épocas
   - Implementar early stopping basado en validación
   - Monitorear curvas de aprendizaje para detectar overfitting

3. **Optimizaciones para ResNet**
   - Revisar implementación de ResNet-18 Base
   - Ajustar learning rate específicamente para familia ResNet
   - Considerar warm-up learning rate schedule más largo

4. **Mejoras en Pipeline de Entrenamiento**
   - Implementar manejo robusto de errores durante entrenamiento
   - Agregar checkpoints automáticos para recovery
   - Mejorar logging y monitoreo de métricas

5. **Análisis Extendido**
   - Generar curvas de entrenamiento para todos los modelos exitosos
   - Implementar análisis de matriz de confusión detallada
   - Evaluar en dataset de test independiente una vez completado el entrenamiento

### 📊 Métricas de Eficiencia

| Modelo | Parámetros Aprox. | Tiempo/Época | Memoria GPU |
|--------|-------------------|--------------|-------------|
| EfficientNet-B0 | 5.3M | Rápido | Bajo |
| EfficientNet-B1 | 7.8M | Medio | Medio |
| EfficientNet-B2 | 9.2M | Medio-Alto | Alto |
| DenseNet-121 | 8.0M | Medio | Medio |

---

### 👥 Información del Grupo

**Grupo 3**  
**Proyecto**: Clasificación de Aeronaves con Deep Learning  
**Fecha**: Agosto 2025  
**Estado**: Resultados Preliminares - 3 Épocas + Análisis Técnico de Fallas en 30 Épocas

---

*Este reporte presenta resultados exitosos de entrenamiento preliminar (3 épocas) y análisis detallado de las fallas técnicas encontradas durante los intentos de entrenamiento extendido (30 épocas). Los mejores modelos (EfficientNet-B1, B2, DenseNet-121) demostraron excelente rendimiento inicial que justifica la resolución de los problemas técnicos para entrenamiento completo.*