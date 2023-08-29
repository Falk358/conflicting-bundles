from models.factory import create_model
from models.fnn_pruned import FNN_pruned
from conflicting_bundle import compute_layerwise_batch_entropy
from config import get_config
from data.factory import load_dataset
import sys


import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import metrics
from keras import layers
import numpy as np

def main():
    #tf.debugging.set_log_device_placement(True)
    config = get_config()
    print(config)
    lbe_threshold_lower = config.lbe_threshold
    train_ds, test_ds = load_dataset(config, augment= False)
    model = create_model(config)
    model_trained = train(config, train_ds, model)
    # pruning loop here
    new_layer_count = 0
    lbe = compute_layerwise_batch_entropy(model = model_trained, train_ds = train_ds, train_batch_size = config.batch_size, all_layers = True)
    for index, layer in enumerate(model_trained.layers):
        if index > len(lbe)-1:
            print(f"skipping layer at index {index}")
            continue # we do not have lbe values for all layers (output for example) and need to skip
        if isinstance(layer, tf.keras.layers.Dense):
            if (lbe[index] > lbe_threshold_lower): # throw away layers with lower lbe than lbe_threshold_lower
                print(f"lbe at layer {index}: {lbe[index]}")
                print(f"adding layer {index} to pruned model")
                new_layer_count += 1
    
    model_pruned = FNN_pruned(config,new_layer_count)
    model_pruned_trained = train(config, train_ds, model_pruned)
    loss_original_model, accuracy_original_model, f1_original_model = test(config, test_ds, model_trained)
    print("orginal model finished; testing pruned model")
    loss_pruned, accuracy_pruned, f1_pruned = test(config, test_ds, model_pruned_trained)

    print(f"Original Model: \n   Loss {loss_original_model}\n   Accuracy: {accuracy_original_model}\n   F1-score (micro): {f1_original_model}")
    print(f"Pruned Model: \n   Loss {loss_pruned}\n   Accuracy: {accuracy_pruned}\n   F1-score (micro): {f1_pruned}")
        


# this custom training function is defined to avoid conflicts with the rest of the project
def train(config, train_ds, model):
    epochs = config.epochs
    print(f"num epochs: {epochs}")
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    radam=tfa.optimizers.RectifiedAdam(
        learning_rate=config.learning_rate, 
        epsilon=1e-6,
        weight_decay=1e-2
    )
    for epoch in range(epochs-1):
        print(f"\nStart of epoch {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training= True) 
                loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                radam.apply_gradients(zip(grads, model.trainable_weights))

            if step % 200 == 0: # log 
                loss_float = float(loss_value)
                print(f"training categorical cross entropy loss for batch at step {step}: {loss_float}")                

    return model

def test(config, test_ds, model):
    test_loss = metrics.Mean(name='test_loss')
    print(f"num classes from config: {config.num_classes}")
    test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_f1 = tfa.metrics.F1Score(num_classes = config.num_classes, average = "macro")
    
    # Define test loss function
    test_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Evaluate on test data
    for x_batch_test, y_batch_test in test_ds:
        test_logits = model(x_batch_test, training=False)
        test_loss_value = test_loss_fn(y_batch_test, test_logits)
        y_batch_test_onehot = tf.one_hot(y_batch_test, depth=config.num_classes)

    
        # Update metrics
        test_loss.update_state(test_loss_value)
        test_accuracy.update_state(y_batch_test, test_logits)
        test_f1.update_state(y_batch_test_onehot, test_logits)
    
    # Retrieve and reset metrics
    final_test_loss = test_loss.result().numpy()
    final_test_accuracy = test_accuracy.result().numpy()
    final_test_f1_score = test_f1.result().numpy()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_f1.reset_states()
    return final_test_loss, final_test_accuracy, final_test_f1_score


if __name__=="__main__":
    main()
