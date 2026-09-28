import numpy as np
import tensorflow as tf

hours = np.array([1, 2, 3, 4, 5, 6], dtype=float)

marks = np.array([35, 42, 50, 58, 67, 75], dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)),
                             tf.keras.layers.Dense(1)])

model.compile(optimizer="adam", loss="mse")

model.fit(hours, marks, epochs=1000, verbose=0)

prediction = model.predict(np.array([[35]]))

print(prediction)
