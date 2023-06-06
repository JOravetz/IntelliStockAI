import tensorflow as tf

# Check TensorFlow Version
print("TensorFlow Version: ", tf.__version__)

# Check for GPU availability with TensorFlow
print("GPU devices: ", tf.config.list_physical_devices('GPU'))

# Check for CUDA support by TensorFlow
print("Is built with CUDA: ", tf.test.is_built_with_cuda())

# Check specific CUDA and cuDNN version in TensorFlow
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']

print("CUDA Version: ", cuda_version)
print("CuDNN Version: ", cudnn_version)

# Load mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple sequential model
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Create a basic model instance
model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Test the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# Save the model
model.save('mnist_model')
