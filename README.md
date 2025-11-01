# Deep Learning Notebooks

## 1. Neural Network Basics

### 1.1 Tensors and Activations Functions

[**01-Tensor_Fundamentals.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/01-Tensor_Fundamentals.ipynb)
- Introducción a tensores, operaciones básicas y manejo de GPU
- Implementación de perceptrón lineal con producto matricial
- Estructura de tensores para imágenes (B, C, H, W) y operaciones de reducción

[**02-AF-Linear_Transfer_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/02-AF-Linear_Transfer_Function.ipynb)
- Función de transferencia lineal (Identity Function)
- Análisis de la función y su derivada constante

[**03-AF-Step_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/03-AF-Step_Function.ipynb)
- Función de activación Step: activación binaria basada en umbral
- Visualización y comportamiento de la función escalón

[**04-AF-ReLu_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/04-AF-ReLu_Function.ipynb)
- ReLU: función de activación estándar en redes modernas
- Comportamiento con valores negativos y positivos

[**04.1-AF-Leaky_ReLu_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/04.1-AF-Leaky_ReLu_Function.ipynb)
- Leaky ReLU: solución al problema de neuronas muertas
- Casos de uso en GANs y redes profundas

[**05-AF-Sigmoid_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/05-AF-Sigmoid_Function.ipynb)
- Sigmoid: transformación a rango [0,1] para clasificación binaria
- Problema del gradiente desaparecido

[**06-AF-Softmax_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/06-AF-Softmax_Function.ipynb)
- Softmax para clasificación multiclase
- Conversión de logits a distribución de probabilidades

### 1.2 Feedforward, Error and Backpropagation

[**01-Feedforward_Process.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/01-Feedforward_Process.ipynb)
- Propagación hacia adelante (forward pass)
- Construcción de redes multicapa con PyTorch

[**02-Error_Functions.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02-Error_Functions.ipynb)
- Funciones de error: concepto y tipos
- MSE para regresión, diferencia entre error con signo y absoluto

[**02.1-Error_MSE.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.1-Error_MSE.ipynb)
- Mean Squared Error aplicado a redes neuronales
- Implementación práctica y visualización

[**02.4-Error_Cross_Entropy.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.4-Error_Cross_Entropy.ipynb)
- Binary Cross-Entropy y Cross-Entropy multiclase
- Cálculo con logits vs probabilidades

[**02.6-Optimization.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.6-Optimization.ipynb)
- Optimización con SGD y learning rate
- Ciclo completo: forward, backward, step

[**02.7-Backpropagation.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.7-Backpropagation.ipynb)
- Algoritmo de backpropagation en detalle
- Regla de la cadena y cálculo de gradientes

[**03-MLP-Non-Conv-Classifcation.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/03-MLP-Non-Conv-Classifcation.ipynb)
- MLP sin convoluciones para MNIST
- Demostración de limitaciones: 669,706 parámetros y pérdida de información espacial

## 2. Convolutional Neural Networks

[**01-CNNs-Kernels-Padding-Stride.ipynb**](2-Convolutional%20Neural%20Networks/01-CNNs-Kernels-Padding-Stride.ipynb)
- Fundamentos de convolución: kernels, padding y stride
- Tipos de pooling (MaxPool vs AvgPool) y alternativas modernas

[**02-CNNs-MNIST-Classification.ipynb**](2-Convolutional%20Neural%20Networks/02-CNNs-MNIST-Classification.ipynb)
- CNN completa para clasificación MNIST
- Arquitectura con capas convolucionales, pooling y dropout

[**02-CNNs-MNIST-Greyscale-Classification.ipynb**](2-Convolutional%20Neural%20Networks/02-CNNs-MNIST-Greyscale-Classification.ipynb)
- Entrenamiento completo de CNN en MNIST
- Loop de entrenamiento, evaluación y guardado del modelo

[**02.1-EXTRA-PreProd-Inference.ipynb**](2-Convolutional%20Neural%20Networks/02.1-EXTRA-PreProd-Inference.ipynb)
- Inferencia con modelo guardado en producción
- Preprocesamiento de imágenes externas

[**03-CNN-CIFAR-10-Normalization.ipynb**](2-Convolutional%20Neural%20Networks/03-CNN-CIFAR-10-Normalization.ipynb)
- Normalización de imágenes RGB
- Cálculo manual de media y desviación estándar por canal

[**03.1-CNN-CIFAR-10-Classification.ipynb**](2-Convolutional%20Neural%20Networks/03.1-CNN-CIFAR-10-Classification.ipynb)
- CNN para CIFAR-10 (10 clases a color)
- Visualización de curvas de aprendizaje, detección de plateau y overfitting

[**03.2-CNN-CIFAR-10-Color-Scheduler.ipynb**](2-Convolutional%20Neural%20Networks/03.2-CNN-CIFAR-10-Color-Scheduler.ipynb)
- Schedulers de learning rate (StepLR)
- Mitigación de plateau con decaimiento adaptativo

[**03.3-CNN-CIFAR-10-Color-Data-Augment.ipynb**](2-Convolutional%20Neural%20Networks/03.3-CNN-CIFAR-10-Color-Data-Augment.ipynb)
- Data Augmentation: RandomCrop y RandomHorizontalFlip
- Mejora de generalización combinando scheduler y augmentation

[**03.4-CNN-CIFAR-10-Color-Parameter-Tunning_and_GPU.ipynb**](2-Convolutional%20Neural%20Networks/03.4-CNN-CIFAR-10-Color-Parameter-Tunning_and_GPU.ipynb)
- Ajuste de hiperparámetros y optimización de GPU
- TensorBoard para visualización en tiempo real

## Requisitos

Las dependencias del proyecto se encuentran especificadas en `requirements.txt`.

## Uso

Los notebooks están diseñados para ejecutarse secuencialmente dentro de cada sección. Se recomienda seguir el orden numérico.

## Dataset

Los notebooks descargan automáticamente los datasets necesarios en el directorio `./data` durante la primera ejecución.

## Hardware

Los notebooks están configurados para aprovechar aceleración GPU mediante CUDA cuando está disponible, con fallback automático a CPU.

