import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Data: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations?select=random_evals.csv

# Load the data
df = pd.read_csv("Positions_test_2.csv")

# Convert the strings to arrays
df['Array'] = df['Array'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

#shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(df['Array'], df['Evaluation'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Convert the training, validation, and testing data to tensors
X_train = tf.convert_to_tensor(np.stack(X_train.values), dtype=tf.float32)
X_val = tf.convert_to_tensor(np.stack(X_val.values), dtype=tf.float32)
X_test = tf.convert_to_tensor(np.stack(X_test.values), dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(13, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
# loss='mean_squared_logarithmic_error' has given the lowest loss so far, need to see if others can do better. They didn't do better.
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

# Train the model
model.fit(X_train, y_train, batch_size=384, epochs=256, validation_data=(X_val, y_val))


import matplotlib.pyplot as plt

train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

# Create figure and set its background color
fig, ax= plt.subplots()

# Training and validation loss plot
ax.plot(epochs, train_loss, label='Training loss', color='goldenrod', marker='+')
ax.plot(epochs, val_loss, label='Validation loss', color='darkblue', linestyle = '-.')
ax.set_title('Training and validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
ax.set_facecolor("dimgray")
fig.set_facecolor("dimgray")
plt.show()



# Evaluate the model on the test set
model.evaluate(X_test, y_test)