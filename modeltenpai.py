# -*- coding: utf-8 -*-
"""model_tenpai.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y4V2H41sKaA9ckkpkao_lIOtyW51F0kB
"""

from tensorflow import keras
from keras import layers
from layer import mlp
from layer import Patches
from layer import PatchEncoder
from keras.layers import Lambda

def create_mjt_1player(input_shape,mask_shape,num_patches,projection_dim,
                          transformer_layers,num_heads,transformer_units,mlp_head_units,num_classes,hai_dim):
  
    inputs = layers.Input(shape=input_shape)
    input_mask = layers.Input(shape=mask_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(hai_dim)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    
    

    # Create multiple layers of the Transformer block.
    
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1,attention_mask = input_mask)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    # encoded_patches = Lambda(lambda a: a[:,:1], input_shape=[None, 25,16])(encoded_patches)###withCLS
    encoded_patches = Lambda(lambda a: a[:,1:], input_shape=[None, 25,104])(encoded_patches)###noCLS
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.15)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.15)
    # Classify outputs.
    logits = layers.Dense(num_classes,activation='sigmoid')(features)
    # Create the Keras model.
    model = keras.Model(inputs=[inputs,input_mask], outputs=logits)
    return model

def create_mjt_4player(input_shape,mask_shape,num_patches,projection_dim,
                          transformer_layers,num_heads,transformer_units,mlp_head_units,num_classes,hai_dim):
    inputs = layers.Input(shape=input_shape)
    input_mask = layers.Input(shape=mask_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches()(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1,attention_mask = input_mask)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    encoded_patches1 = Lambda(lambda a: a[:, 0], input_shape=[None, 100, 104])(encoded_patches)
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches1)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    # Classify outputs.
    logits1 = layers.Dense(num_classes, activation='sigmoid')(features)
    # Create the Keras model.

    encoded_patches2 = Lambda(lambda a: a[:, 1], input_shape=[None, 100, 104])(encoded_patches)
    # Create a [batch_size, projection_dim] tensor.
    representation2 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches2)
    representation2 = layers.Flatten()(representation2)
    representation2 = layers.Dropout(0.3)(representation2)
    # Add MLP.
    features2 = mlp(representation2, hidden_units=mlp_head_units, dropout_rate=0.3)
    # Classify outputs.
    logits2 = layers.Dense(num_classes, activation='sigmoid')(features2)
    # Create the Keras model.

    encoded_patches3 = Lambda(lambda a: a[:, 2], input_shape=[None, 100, 104])(encoded_patches)
    # Create a [batch_size, projection_dim] tensor.
    representation3 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches3)
    representation3 = layers.Flatten()(representation3)
    representation3 = layers.Dropout(0.3)(representation3)
    # Add MLP.
    features3 = mlp(representation3, hidden_units=mlp_head_units, dropout_rate=0.3)
    # Classify outputs.
    logits3 = layers.Dense(num_classes, activation='sigmoid')(features3)
    # Create the Keras model.

    encoded_patches4 = Lambda(lambda a: a[:, 3], input_shape=[None, 100, 104])(encoded_patches)
    # Create a [batch_size, projection_dim] tensor.
    representation4 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches4)
    representation4 = layers.Flatten()(representation4)
    representation4 = layers.Dropout(0.3)(representation4)
    # Add MLP.
    features4 = mlp(representation4, hidden_units=mlp_head_units, dropout_rate=0.3)
    # Classify outputs.
    logits4 = layers.Dense(num_classes, activation='sigmoid')(features4)
    # Create the Keras model.

    model = keras.Model(inputs=[inputs,input_mask], outputs=[logits1, logits2, logits3, logits4])
    return model
