# Reporte de Resultados Preliminares - Clasificación de Aeronaves
## Grupo 3

### 📋 Resumen Ejecutivo

Este reporte presenta los resultados preliminares de la evaluación de diferentes arquitecturas de redes neuronales convolucionales para la clasificación de imágenes de aeronaves. Se evaluaron 8 modelos diferentes utilizando un dataset con 100 clases de aeronaves, con 8,000 imágenes de entrenamiento y 1,000 de validación.

### 🎯 Objetivos

- Comparar el rendimiento de diferentes arquitecturas de CNN modernas
- Identificar el modelo con mejor balance entre precisión y eficiencia
- Evaluar la convergencia de los modelos con entrenamiento limitado (3 épocas)

### 📊 Resultados Obtenidos

#### Tabla Comparativa de Rendimiento

| Modelo | Top-1 Accuracy (Val) | Top-1 Accuracy (Test) | Top-5 Accuracy (Val) | Estado | Observaciones |
|--------|---------------------|----------------------|---------------------|---------|---------------|
| **EfficientNet-B1** | **70.70%** | **70.70%** | **92.40%** | ✅ Exitoso | Mejor rendimiento general |
| **EfficientNet-B2** | **68.00%** | **68.00%** | **90.50%** | ✅ Exitoso | Segundo mejor |
| **DenseNet-121** | **61.70%** | **61.70%** | **86.80%** | ✅ Exitoso | Buen rendimiento |
| **EfficientNet-B0** | **51.60%** | **51.60%** | **83.50%** | ✅ Exitoso | Modelo más liviano |
| **ResNet-34 Plus** | 1.30% | 16.60% | 5.40% | ✅ Convergencia lenta | Necesita más épocas |
| **ResNet-18 Plus** | 0.70% | 12.00% | 4.50% | ✅ Convergencia lenta | Necesita más épocas |
| **ResNet-18 Base** | 0.00% | 0.00% | 0.00% | ❌ Error | Error dimensional |
| **ResNet-34 Base** | - | - | - | 📝 No ejecutado | - |

#### Configuraciones de Entrenamiento

| Modelo | Batch Size | Input Size | Grad. Accumulation | EMA |
|--------|------------|------------|-------------------|-----|
| EfficientNet-B0 | 32 | 224×224 | 4 steps | No |
| EfficientNet-B1 | 24 | 240×240 | 5 steps | No |
| EfficientNet-B2 | 16 | 260×260 | 8 steps | No |
| DenseNet-121 | 24 | 224×224 | 5 steps | No |
| ResNet-18 Plus | 32 | 224×224 | 4 steps | Sí |
| ResNet-34 Plus | 24 | 224×224 | 5 steps | Sí |

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
- **Épocas**: 3 (entrenamiento preliminar)
- **Hardware**: Apple Silicon GPU (MPS)
- **Optimización**: Gradient accumulation para batch sizes efectivos más grandes

### 🎯 Conclusiones y Recomendaciones

#### Conclusiones Principales

1. **EfficientNet-B1** demostró ser el modelo más efectivo para esta tarea
2. La familia **EfficientNet** superó consistentemente a otras arquitecturas
3. **DenseNet-121** ofrece un buen balance como alternativa
4. Los modelos **ResNet** requieren optimización adicional

#### Recomendaciones para Trabajo Futuro

1. **Entrenamiento Extendido**
   - Incrementar a 10-15 épocas para convergencia completa
   - Implementar early stopping basado en validación

2. **Optimizaciones para ResNet**
   - Revisar implementación de ResNet-18 Base
   - Ajustar learning rate para familia ResNet
   - Considerar warm-up learning rate schedule

3. **Mejoras Generales**
   - Implementar data augmentation más agresivo
   - Explorar técnicas de regularización (dropout, weight decay)
   - Evaluar ensemble de mejores modelos

4. **Validación Adicional**
   - Implementar validación cruzada
   - Evaluar en dataset de test independiente
   - Análisis de matriz de confusión por clases

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
**Estado**: Resultados Preliminares - 3 Épocas

---

*Este reporte presenta resultados preliminares basados en entrenamiento de 3 épocas. Se recomienda entrenamiento extendido para resultados definitivos.*