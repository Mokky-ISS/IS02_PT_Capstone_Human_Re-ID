# refer to https://www.tensorflow.org/guide/gpu
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model

tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print("gpus", gpus)
print("logical_gpus", logical_gpus)
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

if gpus:
    try:
        print("test")
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        print("test")
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print("error!")
        print(e)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

with mirrored_strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(renorm=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
