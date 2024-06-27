import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from models.siamese_model import make_siamese_model

# Define the binary cross-entropy loss
binary_cross_loss = tf.losses.BinaryCrossentropy()
# Define the Adam optimizer
opt = tf.keras.optimizers.Adam(1e-4)

# Define a function for a single training step
@tf.function
def train_step(batch, siamese_model):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Reshape y to match yhat's shape
        y = tf.reshape(y, tf.shape(yhat))
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(f"Loss: {loss}")

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # Apply gradients
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss

# Define the training function
def run(train_data, test_data):
    # Make the Siamese model
    siamese_model = make_siamese_model()
    print(siamese_model.summary())

    # Calculate number of batches
    train_steps_per_epoch = sum(1 for _ in train_data)
    test_steps_per_epoch = sum(1 for _ in test_data)

    # Loop through epochs
    for epoch in range(1, 51):  # 50 epochs
        print(f"\nEpoch {epoch}/50")
        progbar = tf.keras.utils.Progbar(train_steps_per_epoch)

        # Create metric objects
        r = Recall()
        p = Precision()

        # Loop through each batch in the dataset
        for idx, batch in enumerate(train_data):
            try:
                # Run the train step here
                loss = train_step(batch, siamese_model)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            except Exception as e:
                print(f"Error processing batch {idx}: {e}")

        print(f"Loss: {loss.numpy()}, Recall: {r.result().numpy()}, Precision: {p.result().numpy()}")

    # Evaluate the model on the test data
    print("\nEvaluating on test data:")
    for test_input, test_val, y_true in test_data:
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true, yhat)
    print(f"Recall: {r.result().numpy()}, Precision: {p.result().numpy()}")

if __name__ == "__main__":
    from scripts.preprocess import run as preprocess_run
    anchor, positive, negative, train_data, test_data = preprocess_run()
    run(train_data, test_data)
