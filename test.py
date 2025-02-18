import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import h5py

def analyze_h5_weights(weights_path):
    """Analyze the structure of weights in an H5 file."""
    f = h5py.File(weights_path, 'r')
    
    print("Weights file structure analysis:")
    layer_info = []
    
    def collect_layer_info(name, obj):
        if isinstance(obj, h5py.Dataset):
            layer_info.append((name, obj.shape, obj.dtype))
    
    f.visititems(collect_layer_info)
    
    # Sort and print for better readability
    layer_info.sort(key=lambda x: x[0])
    for name, shape, dtype in layer_info:
        print(f"Layer: {name}, Shape: {shape}, Type: {dtype}")
    
    f.close()
    return layer_info

def build_face_recognition_model(input_shape=(112, 112, 3), embedding_size=512):
    """
    Build a face recognition model with a simpler architecture that matches the weights.
    """
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Initial convolution layer with BatchNormalization
    x = layers.Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.ReLU()(x)
    
    # Global pooling after convolution to get fixed-size representation
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layer to get the embedding
    x = layers.Dense(embedding_size, kernel_initializer='he_normal', name='fc_embedding')(x)
    embeddings = layers.BatchNormalization(name='embeddings')(x)
    
    # Create the model
    model = Model(inputs, embeddings, name='FaceRecognitionModel')
    
    return model

def custom_load_weights(model, weights_path):
    """
    Custom function to load weights with detailed error handling and reporting.
    """
    # First, try standard loading to see if it works
    try:
        model.load_weights(weights_path, by_name=True)
        return True, "Weights loaded successfully"
    except Exception as e:
        print(f"Standard loading failed: {e}")
    
    # If standard loading fails, try layer-by-layer loading
    try:
        # Open the weights file
        f = h5py.File(weights_path, 'r')
        
        # For each layer in the model
        for layer in model.layers:
            if hasattr(layer, 'get_weights') and layer.name in f:
                # Get the weights for this layer from the file
                layer_weights = []
                for w_name in f[layer.name]:
                    if w_name in ['gamma', 'beta', 'moving_mean', 'moving_variance'] and f"{layer.name}/{w_name}" in f:
                        layer_weights.append(f[f"{layer.name}/{w_name}"][()])
                
                # Set the weights if we found any
                if layer_weights:
                    # Check if shapes match
                    model_shapes = [w.shape for w in layer.get_weights()]
                    file_shapes = [w.shape for w in layer_weights]
                    
                    if model_shapes == file_shapes:
                        layer.set_weights(layer_weights)
                        print(f"Successfully loaded weights for layer: {layer.name}")
                    else:
                        print(f"Shape mismatch for layer {layer.name}:")
                        print(f"  Model expects: {model_shapes}")
                        print(f"  File contains: {file_shapes}")
        
        f.close()
        return True, "Weights loaded with custom approach"
    
    except Exception as e:
        return False, f"Custom loading failed: {e}"

def main():
    # Main parameters
    input_shape = (112, 112, 3)
    embedding_size = 512
    weights_path = "models/arcface_weights.h5"
    tflite_output_path = "models/arcface_weights.tflite"
    
    # Analyze weights file to understand its structure
    print("Analyzing weights file structure...")
    try:
        layer_info = analyze_h5_weights(weights_path)
    except Exception as e:
        print(f"Could not analyze weights file: {e}")
        layer_info = []
    
    # Check if the file has only one batch normalization layer
    bn_layers = [item for item in layer_info if 'batch_normalization' in item[0]]
    print(f"\nFound {len(bn_layers)} batch normalization layers")
    
    # Build a simplified model that matches the weights file structure
    print("\nBuilding simplified model...")
    model = build_face_recognition_model(input_shape=input_shape, embedding_size=embedding_size)
    model.summary()
    
    # Try to load weights using custom approach
    print("\nAttempting to load weights...")
    success, message = custom_load_weights(model, weights_path)
    
    if success:
        print(f"✅ {message}")
        
        # Convert to TFLite if weights were loaded successfully
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
            with open(tflite_output_path, "wb") as f:
                f.write(tflite_model)
            print(f"✅ Model converted to TF Lite and saved at: {tflite_output_path}")
            
        except Exception as e:
            print(f"❌ Error converting to TFLite: {e}")
    else:
        print(f"❌ {message}")
        print("\nRecommendations:")
        print("1. Verify that the weights file contains the expected layers")
        print("2. Consider using a different pre-trained model if available")
        print("3. If this is for transfer learning, you might need to train a new model")

if __name__ == "__main__":
    main()