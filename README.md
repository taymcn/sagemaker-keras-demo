Amazon SageMaker is a modular, fully managed Machine Learning service that lets you easily build, train and deploy models at any scale. https://aws.amazon.com/sagemaker/

In this demo, we demonstrate using SageMaker's script mode, managed spot training, Debugger, Automatic Model Tuning, Experiments, and Model Monitor features. 

We'll use Keras with the TensorFlow backend to build a simple Convolutional Neural Network (CNN) on Amazon SageMaker and train it to classify the Fashion-MNIST image data set. 

Fashion-MNIST is a Zalando dataset consisting of a training set of 60,000 examples and a validation set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes: it's a drop-in replacement for MNIST.

Demo modified from AIM410R/R1 session at AWS re:Invent 2019. https://gitlab.com/juliensimon/aim410

The notebook supports three different versions of the Keras script
* mnist_keras_tf.py: Keras in symbolic mode with TensorFlow 1.15
* mnist_keras_tf20_compat.py: Keras in symbolic mode with TensorFlow 2.0 
* mnist_keras_tf20_eager.py: Keras in eager mode with TensorFlow 2.0

You only need to set the correct TensorFlow version when configuring the TensorFlow estimator.
