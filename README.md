# Deep Learning Notebooks

## 1. Neural Network Basics

### 1.1 Tensors and Activations Functions

[**01-Tensor_Fundamentals.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/01-Tensor_Fundamentals.ipynb)
- Introduction to tensors, basic operations and GPU handling
- Linear perceptron implementation with matrix product
- Tensor structure for images (B, C, H, W) and reduction operations

[**02-AF-Linear_Transfer_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/02-AF-Linear_Transfer_Function.ipynb)
- Linear transfer function (Identity Function)
- Function analysis and its constant derivative

[**03-AF-Step_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/03-AF-Step_Function.ipynb)
- Step activation function: threshold-based binary activation
- Visualization and behavior of the step function

[**04-AF-ReLu_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/04-AF-ReLu_Function.ipynb)
- ReLU: standard activation function in modern networks
- Behavior with negative and positive values

[**04.1-AF-Leaky_ReLu_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/04.1-AF-Leaky_ReLu_Function.ipynb)
- Leaky ReLU: solution to the dying neurons problem
- Use cases in GANs and deep networks

[**05-AF-Sigmoid_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/05-AF-Sigmoid_Function.ipynb)
- Sigmoid: transformation to [0,1] range for binary classification
- Vanishing gradient problem

[**06-AF-Softmax_Function.ipynb**](1-Neural%20Network%20Basics/01-Tensors%20and%20Activations%20Functions/06-AF-Softmax_Function.ipynb)
- Softmax for multiclass classification
- Logits to probability distribution conversion

### 1.2 Feedforward, Error and Backpropagation

[**01-Feedforward_Process.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/01-Feedforward_Process.ipynb)
- Forward propagation (forward pass)
- Building multilayer networks with PyTorch

[**02-Error_Functions.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02-Error_Functions.ipynb)
- Error functions: concept and types
- MSE for regression, difference between signed and absolute error

[**02.1-Error_MSE.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.1-Error_MSE.ipynb)
- Mean Squared Error applied to neural networks
- Practical implementation and visualization

[**02.4-Error_Cross_Entropy.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.4-Error_Cross_Entropy.ipynb)
- Binary Cross-Entropy and multiclass Cross-Entropy
- Calculation with logits vs probabilities

[**02.6-Optimization.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.6-Optimization.ipynb)
- Optimization with SGD and learning rate
- Complete cycle: forward, backward, step

[**02.7-Backpropagation.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/02.7-Backpropagation.ipynb)
- Backpropagation algorithm in detail
- Chain rule and gradient calculation

[**03-MLP-Non-Conv-Classifcation.ipynb**](1-Neural%20Network%20Basics/02-Feedforward,%20Error%20and%20Backpropagation/03-MLP-Non-Conv-Classifcation.ipynb)
- MLP without convolutions for MNIST
- Demonstration of limitations: 669,706 parameters and spatial information loss

## 2. Convolutional Neural Networks

[**01-CNNs-Kernels-Padding-Stride.ipynb**](2-Convolutional%20Neural%20Networks/01-CNNs-Kernels-Padding-Stride.ipynb)
- Convolution fundamentals: kernels, padding and stride
- Pooling types (MaxPool vs AvgPool) and modern alternatives

[**02-CNNs-MNIST-Classification.ipynb**](2-Convolutional%20Neural%20Networks/02-CNNs-MNIST-Classification.ipynb)
- Complete CNN for MNIST classification
- Architecture with convolutional layers, pooling and dropout

[**02-CNNs-MNIST-Greyscale-Classification.ipynb**](2-Convolutional%20Neural%20Networks/02-CNNs-MNIST-Greyscale-Classification.ipynb)
- Complete CNN training on MNIST
- Training loop, evaluation and model saving

[**02.1-EXTRA-PreProd-Inference.ipynb**](2-Convolutional%20Neural%20Networks/02.1-EXTRA-PreProd-Inference.ipynb)
- Inference with saved model in production
- External image preprocessing

[**03-CNN-CIFAR-10-Normalization.ipynb**](2-Convolutional%20Neural%20Networks/03-CNN-CIFAR-10-Normalization.ipynb)
- RGB image normalization
- Manual calculation of mean and standard deviation per channel

[**03.1-CNN-CIFAR-10-Classification.ipynb**](2-Convolutional%20Neural%20Networks/03.1-CNN-CIFAR-10-Classification.ipynb)
- CNN for CIFAR-10 (10 color classes)
- Learning curves visualization, plateau and overfitting detection

[**03.2-CNN-CIFAR-10-Color-Scheduler.ipynb**](2-Convolutional%20Neural%20Networks/03.2-CNN-CIFAR-10-Color-Scheduler.ipynb)
- Learning rate schedulers (StepLR)
- Plateau mitigation with adaptive decay

[**03.3-CNN-CIFAR-10-Color-Data-Augment.ipynb**](2-Convolutional%20Neural%20Networks/03.3-CNN-CIFAR-10-Color-Data-Augment.ipynb)
- Data Augmentation: RandomCrop and RandomHorizontalFlip
- Generalization improvement combining scheduler and augmentation

[**03.4-CNN-CIFAR-10-Color-Parameter-Tunning_and_GPU.ipynb**](2-Convolutional%20Neural%20Networks/03.4-CNN-CIFAR-10-Color-Parameter-Tunning_and_GPU.ipynb)
- Hyperparameter tuning and GPU optimization
- TensorBoard for real-time visualization

[**03.5-CNN-CIFAR-10-Color-More-Tunning.ipynb**](2-Convolutional%20Neural%20Networks/03.5-CNN-CIFAR-10-Color-More-Tunning.ipynb)
- Advanced optimization with ReduceLROnPlateau and label smoothing
- Additional regularization techniques to improve generalization
- Multiclass metrics analysis (Precision, Recall, F1)

[**04-VGGNET16-BN_CIFAR10.ipynb**](2-Convolutional%20Neural%20Networks/04-VGGNET16-BN_CIFAR10.ipynb)
- VGG16 implementation with Batch Normalization for CIFAR-10
- Architecture based on repetitive convolution blocks
- Performance comparison with simple CNNs (~15M parameters)

[**05-ResNet20-CIFAR10.ipynb**](2-Convolutional%20Neural%20Networks/05-ResNet20-CIFAR10.ipynb)
- ResNet-20: introduction to residual connections (skip connections)
- Solution to the vanishing gradient problem in deep networks
- Efficient architecture with 9 residual blocks and ~270K parameters
- MultiStepLR scheduler for optimal convergence

## Requirements

Project dependencies are specified in `requirements.txt`.

## Usage

Notebooks are designed to be executed sequentially within each section. It is recommended to follow the numerical order.

## Dataset

Notebooks automatically download the required datasets to the `./data` directory during the first execution.

## Hardware

Notebooks are configured to leverage GPU acceleration via CUDA when available, with automatic fallback to CPU.

