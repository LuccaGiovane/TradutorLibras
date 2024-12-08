import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

#path pro dataset
base_dir = "../dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

#Letras válidas (A até Y, excluindo H, J, K, X, Z pois nao tiham elas no dataset da kaggle kkk)
valid_letters = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in ['H','J','K','X','Z'] and letter <= 'Y']

# Parâmetros
img_height = 64
img_width = 64
batch_size = 32
epochs = 20

#cria datasets de treino e teste
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    class_names=valid_letters
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    class_names=valid_letters
)

# Otimizações de performance
AUTOTUNE = tf.data.AUTOTUNE

#divisão do conjunto de validação a partir do train_ds
val_size = 0.2
train_size = int((1 - val_size) * tf.data.experimental.cardinality(train_ds).numpy())
val_ds = train_ds.skip(train_size)
train_ds = train_ds.take(train_size)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(valid_letters)

#data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

#modelo CNN
model = models.Sequential([
    #data augmentation
    data_augmentation,
    layers.Rescaling(1./255),
    
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

#avaliação final no conjunto de teste
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nDesempenho no Teste - Loss: {test_loss:.4f} - Accuracy: {test_acc:.4f}")

#salva o modelo treinado
model.save("model.h5")
print("Treinamento concluído e modelo salvo em model.h5.")
