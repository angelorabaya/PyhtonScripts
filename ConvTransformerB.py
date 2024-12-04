import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load and preprocess data
def load_data(file_path):
    # Read CSV with the correct column name
    df = pd.read_csv(file_path)

    # Rename 'Local time' to 'Date' if needed
    if 'Local time' in df.columns:
        df = df.rename(columns={'Local time': 'Date'})

    # Convert to datetime with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')

    # Sort by date
    df = df.sort_values('Date')

    # Resample to hourly frequency if there are any gaps
    df = df.set_index('Date')
    df = df.resample('h').ffill()
    df = df.reset_index()

    return df


# Rest of the code remains unchanged
def prepare_data(data, look_back_hours=24 * 7, future_hours=24 * 3):  # 7 days look back, 3 days forecast
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

    X, y = [], []
    for i in range(len(scaled_data) - look_back_hours - future_hours + 1):
        X.append(scaled_data[i:(i + look_back_hours)])
        y.append(
            scaled_data[i + look_back_hours:i + look_back_hours + future_hours, 3])  # Predicting future Close prices

    return np.array(X), np.array(y), scaler


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def build_conv_transformer(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0,
                           mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs

    # Multiple Convolutional layers
    x = Conv1D(filters=head_size, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters=head_size, kernel_size=5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], head_size)
    x = x + pos_encoding

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ffn_output = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Conv1D(filters=head_size, kernel_size=1)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # MLP layers with residual connections
    for dim in mlp_units:
        skip = x
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
        if dim == mlp_units[0]:  # Add residual connection for the first dense layer
            x = Add()([x, Dense(dim)(skip)])

    outputs = Dense(24 * 3)(x)  # 72 hours (3 days) forecast

    return Model(inputs, outputs)


def predict_future(model, last_sequence, scaler, num_hours=24 * 3):
    future_predictions = model.predict(last_sequence.reshape(1, *last_sequence.shape))[0]
    future_predictions_reshaped = np.zeros((future_predictions.shape[0], 5))
    future_predictions_reshaped[:, 3] = future_predictions
    return scaler.inverse_transform(future_predictions_reshaped)[:, 3]


def main(file_path):
    # Load and prepare data
    df = load_data(file_path)
    look_back_hours = 24 * 7  # 7 days
    future_hours = 24 * 3  # 3 days
    X, y, scaler = prepare_data(df, look_back_hours, future_hours)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Build the model with modified parameters
    model = build_conv_transformer(
        input_shape=X.shape[1:],
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[256, 128],
        mlp_dropout=0.4,
        dropout=0.25
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae"])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=7,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Predict future prices
    last_sequence = X[-1]
    future_prices = predict_future(model, last_sequence, scaler, num_hours=future_hours)

    # Generate future dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                 periods=future_hours,
                                 freq='H')

    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })

    print("\nFuture price predictions (next 72 hours):")
    print(predictions_df)


if __name__ == "__main__":
    file_path = "ADAUSDH.csv"
    main(file_path)