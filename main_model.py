import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
#используйте 2 строчки ниже, если у вас видеокарта от nvidia
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
batch_size = 16
img_size = (512, 512)

# Напишите свой путь до датасета в переменных ниже
val_dir = r"C:\Users\spect\Downloads\chest_xray\train"
train_dir = r"C:\Users\spect\Downloads\chest_xray\train"
test_dir=r"C:\Users\spect\Downloads\chest_xray\val"

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed = 42,
    batch_size=batch_size,
    image_size =img_size)
validation_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed = 42,
    batch_size=batch_size,
    image_size =img_size)
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed = 42,
    batch_size=batch_size,
    image_size=img_size)

class_names_num = len(train_data.class_names)

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=1./255),
    
    tf.keras.layers.Conv2D(16, (5,5), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(32, (5,5), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(class_names_num, activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data, validation_data=validation_data, epochs=3)

model.evaluate(test_data)

model.save("assets/model_files/model.h5")