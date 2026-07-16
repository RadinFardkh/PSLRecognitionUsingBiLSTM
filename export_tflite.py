import tensorflow as tf
import numpy as np
import os
import config
import json

# Load samples for calibration
with open(os.path.join(config.PROCESSED_DIR, "samples.json")) as f:
    samples = json.load(f)
# Use a subset (e.g., 200) as representative dataset
rep_samples = samples[:200]  # take first 200


def representative_dataset_gen():
    for s in rep_samples:
        seq = np.load(os.path.join(config.PROCESSED_DIR, s["file"])).astype(np.float32)
        # Pad/truncate to SEQUENCE_LENGTH
        if len(seq) > config.SEQUENCE_LENGTH:
            seq = seq[:config.SEQUENCE_LENGTH]
        else:
            pad_len = config.SEQUENCE_LENGTH - len(seq)
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
        seq = np.expand_dims(seq, axis=0).astype(np.float32)
        yield [seq]


# Load trained Keras model
model = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, 'final_model.keras'))
# Alternatively load the best checkpoint in case of crash:
# model = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, 'best_model.h5'))

# Converts the model to tflite for Android compatibility
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if config.QUANTIZE:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS  # <-- added
    ]
    converter._experimental_lower_tensor_list_ops = False  # <-- added
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_path = os.path.join(config.MODEL_DIR, config.TFLITE_MODEL_NAME)
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to {tflite_path}")
