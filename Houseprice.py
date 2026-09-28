#Houseprice
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ==========================================
# 1. NUMPY: Generate Raw Numerical Data
# ==========================================
print("--- Step 1: Generating Data with NumPy ---")
np.random.seed(42)  # For reproducibility

# Generate 1000 samples: Size of house (sq ft) and Number of bedrooms
house_sizes = np.random.randint(800, 3500, size=1000)
bedrooms = np.random.randint(1, 6, size=1000)

# Calculate a target price with some random noise added
prices = (house_sizes * 150) + (bedrooms * 10000) + np.random.normal(0, 5000, size=1000)


# ==========================================
# 2. PANDAS: Organize and Clean the Data
# ==========================================
print("\n--- Step 2: Structuring Data with Pandas ---")

# Create a DataFrame
dataset = pd.DataFrame({
    'House_Size': house_sizes,
    'Bedrooms': bedrooms,
    'Price': prices
})

# Display the first 5 rows
print("Dataset Head:")
print(dataset.head())

# Split features (X) and target (y)
X_df = dataset[['House_Size', 'Bedrooms']]
y_df = dataset['Price']

# Convert Pandas DataFrames directly into NumPy arrays for the model
X = X_df.values
y = y_df.values


# ==========================================
# 3. TENSORFLOW: Build and Train a Neural Network
# ==========================================
print("\n--- Step 3: Training Model with TensorFlow ---")

# Define a simple Sequential Neural Network model
model = tf.keras.Sequential([
    layers.Input(shape=(2,)),                  # Input layer expecting 2 features
    layers.Dense(64, activation='relu'),       # Hidden layer 1
    layers.Dense(32, activation='relu'),       # Hidden layer 2
    layers.Dense(1)                            # Output layer predicting 1 continuous price value
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',                                # Mean Squared Error for regression
    metrics=['mae']                            # Mean Absolute Error tracking
)

# Train the model
print("Training the neural network...")
history = model.fit(X, y, epochs=20, batch_size=32, verbose=1, validation_split=0.2)


# ==========================================
# 4. PREDICTION: Test the Model
# ==========================================
print("\n--- Step 4: Making a Prediction ---")

# Predict the price of a new house (2000 sq ft, 3 bedrooms)
new_house = np.array([[2000, 3]])
predicted_price = model.predict(new_house)

print(f"Predicted Price for a 2000 sq ft, 3-bed house: ${predicted_price[0][0]:,.2f}")