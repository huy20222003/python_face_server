import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import h5py
import numpy as np

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

def build_advanced_face_recognition_model(input_shape=(112, 112, 3), embedding_size=512):
    """
    Build an advanced face recognition model with attention mechanism and dropout.
    """
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Initial convolution block
    x = layers.Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # First residual block with dropout
    residual = x
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)  # Add dropout
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    residual = layers.Conv2D(128, 1, padding='same')(residual)
    x = layers.add([x, residual])
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Second residual block with dropout
    residual = x
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)  # Add dropout
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    residual = layers.Conv2D(256, 1, padding='same')(residual)
    x = layers.add([x, residual])
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # Attention mechanism
    attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    # Feature extraction block
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers for embedding with dropout
    x = layers.Dense(embedding_size, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Add dropout
    x = layers.Dense(embedding_size, kernel_initializer='he_normal', name='fc_embedding')(x)
    embeddings = layers.BatchNormalization(name='embeddings')(x)
    
    # Create the model
    model = Model(inputs, embeddings, name='AdvancedFaceRecognitionModel')
    
    return model

def improved_load_weights_with_shape_check(model, weights_path, custom_mappings=None):
    """
    Cải thiện hàm tải trọng số với kiểm tra kích thước tự động và điều chỉnh.
    """
    if custom_mappings is None:
        custom_mappings = {}
    
    # Phân tích tệp trọng số
    f = h5py.File(weights_path, 'r')
    layer_weights = {}
    
    def collect_layer_info(name, obj):
        if isinstance(obj, h5py.Dataset):
            layer_name = name.split('/')[0] if '/' in name else name
            if layer_name not in layer_weights:
                layer_weights[layer_name] = []
            layer_weights[layer_name].append((name, obj.shape, obj.dtype))
    
    f.visititems(collect_layer_info)
    
    # Xây dựng ánh xạ dựa trên tên lớp và kích thước
    loaded_layers = set()
    for layer in model.layers:
        if not hasattr(layer, 'get_weights'):
            continue
        
        model_weights = layer.get_weights()
        if not model_weights:
            continue
        
        # Tìm lớp phù hợp trong tệp trọng số
        matched_layer_name = None
        
        # Ưu tiên ánh xạ tùy chỉnh
        if layer.name in custom_mappings:
            matched_layer_name = custom_mappings[layer.name]
        else:
            # Tìm lớp có tên và kích thước tương đương
            for weight_layer_name, weights_info in layer_weights.items():
                # Kiểm tra tên tương đồng
                if (layer.name == weight_layer_name or 
                    (layer.name.startswith('batch_normalization') and weight_layer_name.startswith('batch_normalization'))):
                    # Kiểm tra kích thước tương đồng
                    if len(model_weights) == len(weights_info):
                        matched_layer_name = weight_layer_name
                        break
        
        if matched_layer_name and matched_layer_name in layer_weights:
            try:
                # Tải trọng số cho lớp này
                loaded_weights = []
                weights_info = layer_weights[matched_layer_name]
                weights_loaded = False
                
                for i, w in enumerate(model_weights):
                    # Chỉ tải nếu kích thước khớp
                    if i < len(weights_info) and w.shape == weights_info[i][1]:
                        weight_dataset = f[weights_info[i][0]]
                        loaded_weights.append(weight_dataset[()])
                        weights_loaded = True
                    else:
                        print(f"Bỏ qua trọng số không khớp kích thước cho {layer.name}: mô hình cần {w.shape}")
                        loaded_weights.append(w)  # Giữ trọng số ban đầu
                
                if weights_loaded:
                    layer.set_weights(loaded_weights)
                    loaded_layers.add(layer.name)
                    print(f"Đã tải trọng số thành công cho lớp: {layer.name}")
            except Exception as e:
                print(f"Lỗi khi tải trọng số cho lớp {layer.name}: {e}")
    
    f.close()
    
    # Báo cáo kết quả
    if loaded_layers:
        print(f"Đã tải trọng số cho {len(loaded_layers)} lớp")
        return True, f"Đã tải trọng số cho {len(loaded_layers)} lớp"
    else:
        return False, "Không có lớp nào được tải"

def evaluate_model_performance(model, test_data=None, batch_size=32):
    """
    Evaluate model performance on test data or generate synthetic test data.
    
    Args:
        model: The face recognition model to evaluate
        test_data: Optional tuple of (x_test, y_test)
        batch_size: Batch size for evaluation
    """
    if test_data is None:
        # Generate synthetic test data - pairs of faces (same/different)
        print("Generating synthetic evaluation data...")
        num_samples = 1000
        x_test = np.random.rand(num_samples, 112, 112, 3).astype(np.float32)
        y_test = np.random.randint(0, 2, size=(num_samples, 1))
        
        # Convert to embeddings
        embeddings = model.predict(x_test, batch_size=batch_size)
        
        # Calculate distances between pairs
        distances = []
        for i in range(0, num_samples, 2):
            if i+1 < num_samples:
                dist = np.linalg.norm(embeddings[i] - embeddings[i+1])
                distances.append(dist)
        
        distances = np.array(distances)
        
        # Find threshold that maximizes accuracy
        best_acc = 0
        best_threshold = 0
        for threshold in np.linspace(np.min(distances), np.max(distances), 100):
            predictions = (distances < threshold).astype(int)
            labels = y_test[::2].flatten()[:len(predictions)]
            accuracy = np.mean(predictions == labels)
            if accuracy > best_acc:
                best_acc = accuracy
                best_threshold = threshold
        
        print(f"Best threshold: {best_threshold:.4f}")
        print(f"Accuracy on synthetic data: {best_acc:.4f}")
    else:
        # Use provided test data
        x_test, y_test = test_data
        
        # For face verification tasks, process pairs
        if isinstance(x_test, tuple) and len(x_test) == 2:
            # Assuming x_test = (faces1, faces2) and y_test = same/different labels
            embeddings1 = model.predict(x_test[0], batch_size=batch_size)
            embeddings2 = model.predict(x_test[1], batch_size=batch_size)
            
            # Calculate distances
            distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
            
            # Find optimal threshold
            best_acc = 0
            best_threshold = 0
            for threshold in np.linspace(np.min(distances), np.max(distances), 100):
                predictions = (distances < threshold).astype(int)
                accuracy = np.mean(predictions == y_test)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_threshold = threshold
            
            print(f"Best threshold: {best_threshold:.4f}")
            print(f"Accuracy on test data: {best_acc:.4f}")

def convert_to_tflite_with_optimization(model, tflite_output_path):
    """
    Convert model to TFLite with improved optimization settings.
    """
    # Default optimization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization for smaller model size and faster inference
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Enable weight quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]  # Cho phép Float16
    converter.inference_input_type = tf.float32  # Input giữ nguyên
    converter.inference_output_type = tf.float32  # Output giữ nguyên

    # Representative dataset for quantization calibration
    def representative_dataset_gen():
        for _ in range(100):
            # Generate random input data for calibration
            input_data = np.random.rand(1, 112, 112, 3).astype(np.float32)
            yield [input_data]
    
    converter.representative_dataset = representative_dataset_gen
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    
    # Calculate and print model size
    model_size_mb = os.path.getsize(tflite_output_path) / (1024 * 1024)
    print(f"TFLite model size: {model_size_mb:.2f} MB")
    
    return tflite_output_path

def main():
    # Các tham số chính
    input_shape = (112, 112, 3)
    embedding_size = 512
    weights_path = "models/arcface_weights.h5"
    tflite_output_path = "models/arcface_model.tflite"
    
    # Phân tích tệp trọng số để hiểu cấu trúc
    print("Đang phân tích cấu trúc tệp trọng số...")
    try:
        layer_info = analyze_h5_weights(weights_path)
    except Exception as e:
        print(f"Không thể phân tích tệp trọng số: {e}")
        layer_info = []
    
    # Xây dựng ánh xạ tùy chỉnh từ kết quả phân tích
    custom_mappings = {}
    
    # Xây dựng mô hình cải tiến
    print("\nĐang xây dựng mô hình nhận diện khuôn mặt nâng cao...")
    model = build_advanced_face_recognition_model(input_shape=input_shape, embedding_size=embedding_size)
    model.summary()
    
    # Thử tải trọng số bằng cách tiếp cận cải tiến
    print("\nĐang thử tải trọng số với bộ tải cải tiến...")
    
    # Tạo ánh xạ tùy chỉnh dựa trên phân tích
    for layer in model.layers:
        if 'batch_normalization' in layer.name:
            for name, shape, _ in layer_info:
                if 'batch_normalization' in name and '/gamma' in name:
                    batch_norm_name = name.split('/')[0]
                    custom_mappings[layer.name] = batch_norm_name
                    break
    
    # Sử dụng hàm tải trọng số cải tiến
    success, message = improved_load_weights_with_shape_check(model, weights_path, custom_mappings)
    
    if success:
        print(f"✅ {message}")
        
        # Evaluate the model performance
        print("\nEvaluating model performance...")
        evaluate_model_performance(model)
        
        # Convert to optimized TFLite
        print("\nConverting to optimized TFLite model...")
        try:
            tflite_path = convert_to_tflite_with_optimization(model, tflite_output_path)
            print(f"✅ Optimized TFLite model saved at: {tflite_path}")
        except Exception as e:
            print(f"❌ Error converting to TFLite: {e}")
            print("Proceeding with standard TFLite conversion...")
            
            # Fallback to standard conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
            with open(tflite_output_path, "wb") as f:
                f.write(tflite_model)
            print(f"✅ Standard TFLite model saved at: {tflite_output_path}")
    else:
        print(f"❌ {message}")
        print("\nRecommendations:")
        print("1. Modify the custom mappings based on the weight file analysis")
        print("2. Try a different pre-trained model architecture")
        print("3. Consider using a model specifically designed for TFLite deployment")
        print("4. If weights cannot be loaded, consider training a new model")

if __name__ == "__main__":
    main()