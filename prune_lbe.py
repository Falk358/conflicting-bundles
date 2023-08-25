from models.factory import create_model
from conflicting_bundle import compute_layerwise_batch_entropy
from config import get_config
from data.factory import load_dataset

import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import layers
import numpy as np

def main():
    config = get_config()
    train_ds, test_ds = load_dataset(config)
    model = create_model(config)
    model_trained = train(config, train_ds, model)
    # pruning loop here
    lbe_threshold_lower = 0.2
    new_layers = []
    lbe = compute_layerwise_batch_entropy(model = model_trained, train_ds = train_ds, all_layers = True)
    for index, layer in enumerate(model_trained.layers):
        if (lbe[i] > lbe_threshold_lower): # throw away layers with lower lbe than lbe_threshold_lower
            new_layers.append(layer)
    
    model_pruned = tf.keras.Sequential(new_layers)
        


# this custom training function is defined to avoid conflicts with the rest of the project
def train(config, train_ds, model):
    epochs = config.epochs
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    radam=tfa.optimizers.RectifiedAdam(
        learning_rate=config.learning_rate, 
        epsilon=1e-6,
        weight_decay=1e-2
    )
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training= True) 
                loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                radam.apply_gradients(zip(grads, model.trainable_weights))

            if step % 200 == 0: # log 
                loss_float = float(loss_value)
                print(f"training categorical cross entropy loss for batch at step {step}: {loss_float}")                

    return model


if __name__=="__main__":
    main()