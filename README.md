# Cuadernos de Deep Learning

## 1. Fundamentos de Redes Neuronales

### 1.1 Tensores y Funciones de Activación

[**01-Tensor_Fundamentals.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/01-Tensor_Fundamentals.ipynb)
- Introducción a los tensores, operaciones básicas y uso de la GPU
- Implementación de un perceptrón lineal mediante producto matricial
- Estructura tensorial para imágenes (B, C, H, W) y operaciones de reducción

[**02-AF-Linear_Transfer_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/02-AF-Linear_Transfer_Function.ipynb)
- Función de transferencia lineal (función identidad)
- Análisis de la función y de su derivada constante

[**03-AF-Step_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/03-AF-Step_Function.ipynb)
- Función de activación escalón: activación binaria basada en umbral
- Visualización y comportamiento de la función escalón

[**04-AF-ReLu_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/04-AF-ReLu_Function.ipynb)
- ReLU como función estándar en redes modernas
- Comportamiento con valores negativos y positivos

[**04.1-AF-Leaky_ReLu_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/04.1-AF-Leaky_ReLu_Function.ipynb)
- Leaky ReLU como solución al problema de neuronas moribundas
- Casos de uso en GANs y redes profundas

[**05-AF-Sigmoid_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/05-AF-Sigmoid_Function.ipynb)
- Sigmoide: transformación al rango [0, 1] para clasificación binaria
- Problema del gradiente que se desvanece

[**06-AF-Softmax_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/06-AF-Softmax_Function.ipynb)
- Softmax para clasificación multiclase
- Conversión de logits a distribución de probabilidad

### 1.2 Feedforward, Error y Backpropagation

[**01-Feedforward_Process.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/01-Feedforward_Process.ipynb)
- Propagación hacia delante (forward pass)
- Construcción de redes multilayer con PyTorch

[**02-Error_Functions.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02-Error_Functions.ipynb)
- Funciones de error: concepto y tipología
- MSE para regresión; diferencia entre error con signo y absoluto

[**02.1-Error_MSE.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.1-Error_MSE.ipynb)
- Error cuadrático medio aplicado a redes neuronales
- Implementación práctica y visualización

[**02.4-Error_Cross_Entropy.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.4-Error_Cross_Entropy.ipynb)
- Binary Cross-Entropy y Cross-Entropy multiclase
- Cálculo con logits frente a probabilidades

[**02.6-Optimization.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.6-Optimization.ipynb)
- Optimización con SGD y tasa de aprendizaje
- Ciclo completo: forward, backward, step

[**02.7-Backpropagation.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.7-Backpropagation.ipynb)
- Algoritmo de backpropagation en detalle
- Regla de la cadena y cálculo de gradientes

[**03-MLP-Non-Conv-Classifcation.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/03-MLP-Non-Conv-Classifcation.ipynb)
- MLP sin convoluciones para MNIST
- Limitaciones demostradas: 669 706 parámetros y pérdida de información espacial

## 2. Redes Neuronales Convolucionales

[**01-CNNs-Kernels-Padding-Stride.ipynb**](2-Convolutional%20Neural%20Networks/01-CNNs-Kernels-Padding-Stride.ipynb)
- Fundamentos de convoluciones: kernels, padding y stride
- Tipos de pooling (MaxPool vs AvgPool) y alternativas modernas

[**02-CNNs-MNIST-Classification.ipynb**](2-Convolutional%20Neural%20Networks/02-CNNs-MNIST-Classification.ipynb)
- CNN completa para clasificación de MNIST
- Arquitectura con capas convolucionales, pooling y dropout

[**02-CNNs-MNIST-Greyscale-Classification.ipynb**](2-Convolutional%20Neural%20Networks/02-CNNs-MNIST-Greyscale-Classification.ipynb)
- Entrenamiento completo de una CNN sobre MNIST
- Bucle de entrenamiento, evaluación y guardado del modelo

[**02.1-EXTRA-PreProd-Inference.ipynb**](2-Convolutional%20Neural%20Networks/02.1-EXTRA-PreProd-Inference.ipynb)
- Inferencia con un modelo guardado en producción
- Preprocesamiento de imágenes externas

[**03-CNN-CIFAR-10-Normalization.ipynb**](2-Convolutional%20Neural%20Networks/03-CNN-CIFAR-10-Normalization.ipynb)
- Normalización de imágenes RGB
- Cálculo manual de la media y desviación estándar por canal

[**03.1-CNN-CIFAR-10-Classification.ipynb**](2-Convolutional%20Neural%20Networks/03.1-CNN-CIFAR-10-Classification.ipynb)
- CNN para CIFAR-10 (10 clases a color)
- Visualización de curvas de aprendizaje, detección de meseta y sobreajuste

[**03.2-CNN-CIFAR-10-Color-Scheduler.ipynb**](2-Convolutional%20Neural%20Networks/03.2-CNN-CIFAR-10-Color-Scheduler.ipynb)
- Planificadores de tasa de aprendizaje (StepLR)
- Mitigación de mesetas con decaimiento adaptativo

[**03.3-CNN-CIFAR-10-Color-Data-Augment.ipynb**](2-Convolutional%20Neural%20Networks/03.3-CNN-CIFAR-10-Color-Data-Augment.ipynb)
- Data augmentation con RandomCrop y RandomHorizontalFlip
- Mejora de la generalización combinando scheduler y augmentación

[**03.4-CNN-CIFAR-10-Color-Parameter-Tunning_and_GPU.ipynb**](2-Convolutional%20Neural%20Networks/03.4-CNN-CIFAR-10-Color-Parameter-Tunning_and_GPU.ipynb)
- Ajuste de hiperparámetros y optimización en GPU
- Uso de TensorBoard para visualización en tiempo real

[**03.5-CNN-CIFAR-10-Color-More-Tunning.ipynb**](2-Convolutional%20Neural%20Networks/03.5-CNN-CIFAR-10-Color-More-Tunning.ipynb)
- Optimización avanzada con ReduceLROnPlateau y label smoothing
- Técnicas adicionales de regularización para mejorar la generalización
- Análisis de métricas multiclase (Precision, Recall, F1)

[**04-VGGNET16-BN_CIFAR10.ipynb**](2-Convolutional%20Neural%20Networks/04-VGGNET16-BN_CIFAR10.ipynb)
- Implementación de VGG16 con Batch Normalization para CIFAR-10
- Arquitectura basada en bloques convolucionales repetitivos
- Comparativa de rendimiento frente a CNN simples (~15 M de parámetros)

[**05-ResNet20-CIFAR10.ipynb**](2-Convolutional%20Neural%20Networks/05-ResNet20-CIFAR10.ipynb)
- ResNet-20 e introducción a las conexiones residuales (skip connections)
- Solución al problema del gradiente que se desvanece en redes profundas
- Arquitectura eficiente con 9 bloques residuales y ~270 K parámetros
- Scheduler MultiStepLR para convergencia óptima

## 3. Transfer Learning

[**01-Pretrained Network as Feature Extractor.ipynb**](3-Transfer%20Learning/01-Pretrained%20Network%20as%20Feature%20Extractor.ipynb)
- Congelación de la VGG16 preentrenada en ImageNet para extracción de características
- Construcción de un clasificador personalizado para un conjunto de datos reducido
- Normalización con estadísticas estándar de ImageNet y pipeline de preprocesado con `transforms`
- Preparación de datasets con `ImageFolder` y carga con `DataLoader` para `train`, `valid` y `test`

## Requisitos

Las dependencias del proyecto están especificadas en `requirements.txt`.

## Uso

Los cuadernos están diseñados para ejecutarse de forma secuencial en cada sección. Se recomienda seguir el orden numérico.

## Conjuntos de Datos

Los cuadernos descargan automáticamente los datasets necesarios en el directorio `./data` durante la primera ejecución.

## Hardware

Los cuadernos están configurados para aprovechar la aceleración por GPU mediante CUDA cuando está disponible, con un cambio automático a CPU en caso contrario.

