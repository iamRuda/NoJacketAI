import tensorflow as tf

# Печать списка доступных устройств
print("Available devices:", tf.config.list_physical_devices())

# Печать информации о GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("Name:", gpu.name, "Type:", gpu.device_type)
else:
    print("No GPU devices found.")