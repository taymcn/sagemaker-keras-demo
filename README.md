Demo modified from AIM410R/R1 session at AWS re:Invent 2019. https://gitlab.com/juliensimon/aim410

The notebook supports three different versions of the Keras script
* mnist_keras_tf.py: Keras in symbolic mode with TensorFlow 1.15
* mnist_keras_tf20_compat.py: Keras in symbolic mode with TensorFlow 2.0 
* mnist_keras_tf20_eager.py: Keras in eager mode with TensorFlow 2.0

You only need to set the correct TensorFlow version when configuring the TensorFlow estimator.
