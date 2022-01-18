import sys
import tensorflow as tf
import numpy


def main():
    if len(sys.argv) != 2:
        print("Use like this: python training.py [file location to store model]")
        return

    # Use mnist database
    mnist = tf.keras.datasets.mnist

    # Arrange data. x being images and y being labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    for image_num, image in enumerate(x_train):
        # For each image
        for row_num, row in enumerate(image):
            for column_num, item in enumerate(row):
                if item != [0.]:
                    x_train[image_num, row_num, column_num] = [1.]

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    for image_num, image in enumerate(x_test):
        # For each image
        for row_num, row in enumerate(image):
            for column_num, item in enumerate(row):
                if item != [0.]:
                    x_test[image_num, row_num, column_num] = [1.]

    model = build_model()

    # Train model
    model.fit(x_train, y_train, epochs=10)

    # Evaluate model
    model.evaluate(x_test, y_test, verbose=2)

    # Save to file
    file = sys.argv[1]
    model.save(file)
    print("Model Saved.")


def build_model():
    # Choose layers for neural network model
    model = tf.keras.models.Sequential([
        # Convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

        # Max-pooling
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

        # Max-pooling
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten
        tf.keras.layers.Flatten(),

        # Hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),

        # Dropout
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    main()
