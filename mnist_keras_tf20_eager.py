import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import categorical_crossentropy

import numpy as np
import argparse, os, subprocess, sys

# Script mode doesn't support requirements.txt
# Here's the workaround ;)
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

class FMNISTModel(keras.Model):
    # Create the different layers used by the model
    def __init__(self, dense_layer, dropout):
        super(FMNISTModel, self).__init__(name='fmnist_model')
        self.conv2d_1   = Conv2D(64, 3, padding='same', activation='relu',input_shape=(28,28))
        self.conv2d_2   = Conv2D(64, 3, padding='same', activation='relu')
        self.max_pool2d = MaxPooling2D((2, 2), padding='same')
        self.batch_norm = BatchNormalization()
        self.flatten    = Flatten()
        self.dense1     = Dense(dense_layer, activation='relu')
        self.dense2     = Dense(10)
        self.dropout    = Dropout(dropout)
        self.softmax    = Softmax()

    # Chain the layers for forward propagation
    def call(self, x):
        # 1st convolution block
        x = self.conv2d_1(x)
        x = self.max_pool2d(x)
        x = self.batch_norm(x)
        # 2nd convolution block
        x = self.conv2d_2(x)
        x = self.max_pool2d(x)
        x = self.batch_norm(x)
        # Flatten and classify
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.softmax(x)

    
if __name__ == '__main__':

    print("TensorFlow version", tf.__version__)
    print("Keras version", keras.__version__)

    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    import keras_metrics
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dense-layer', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    dense_layer = args.dense_layer
    dropout    = args.dropout
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    # Load data set
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

    # Add extra dimension for channel: (28,28) --> (28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_val   = x_val[..., tf.newaxis]

    # Prepare training and validation iterators
    #  - define batch size
    #  - normalize pixel values to [0,1]
    #  - one-hot encode labels
    preprocess = lambda x, y: (tf.divide(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10)))

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    train = train.map(preprocess)
    train = train.repeat()

    val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    val = val.map(preprocess)
    val = val.repeat()

    # Build model
    model = FMNISTModel(dense_layer, dropout)
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy',
                  keras_metrics.precision(), 
                  keras_metrics.recall(),
                  keras_metrics.f1_score()])
    
    # Train model
    train_steps = x_train.shape[0] / batch_size
    val_steps   = x_val.shape[0] / batch_size

    model.fit(train, epochs=epochs, steps_per_epoch=train_steps, validation_data=val, validation_steps=val_steps)

    score = model.evaluate(val, steps=val_steps)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    
    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))
