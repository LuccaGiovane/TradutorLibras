import os
import threading
import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def iniciar():
    # Executa o script hand_recognition.py para reconhecimento em tempo real das mãos
    # Se quiser manter o subprocess aqui, pode. Ex:
    # subprocess.run(["python", "scripts/hand_recognition.py"])
    pass

def calibrar():
    # Inicia o treinamento em uma thread separada
    train_thread = threading.Thread(target=train_model)
    train_thread.start()

def train_model():
    # Desabilita o botão Calibrar durante o treinamento
    calibrar_button.config(state='disabled')
    progress_var.set(0)
    progress_label.config(text="Treinando... 0%")
    
    # Código de treinamento embutido aqui
    base_dir = "../dataset"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Letras válidas
    valid_letters = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in ['H','J','K','X','Z'] and letter <= 'Y']

    img_height = 64
    img_width = 64
    batch_size = 32
    epochs = 20

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

    AUTOTUNE = tf.data.AUTOTUNE

    val_size = 0.2
    train_size = int((1 - val_size) * tf.data.experimental.cardinality(train_ds).numpy())
    val_ds = train_ds.skip(train_size)
    train_ds = train_ds.take(train_size)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(valid_letters)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
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

    # Callback para atualizar a barra de progresso
    class ProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Atualiza a barra de progresso a cada fim de época
            progresso = int(((epoch+1) / epochs) * 100)
            progress_var.set(progresso)
            progress_label.config(text=f"Treinando... {progresso}%")
            root.update_idletasks()

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, ProgressCallback()]
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nDesempenho no Teste - Loss: {test_loss:.4f} - Accuracy: {test_acc:.4f}")

    model.save("model.h5")
    print("Treinamento concluído e modelo salvo em model.h5.")

    # Atualiza a barra de progresso e feedback final
    progress_var.set(100)
    progress_label.config(text="Treinamento concluído!")
    calibrar_button.config(state='normal')


def main():
    global root, calibrar_button, progress_var, progress_label

    # Interface gráfica principal
    root = tk.Tk()
    root.title("Reconhecimento de Mãos")
    root.configure(bg='#ADD8E6')  # Fundo azul claro

    # Define o tamanho da janela
    root.geometry('400x600')
    root.resizable(False, False)

    # Título
    titulo_font = font.Font(family='Helvetica', size=24, weight='bold')
    titulo_label = tk.Label(root, text="Reconhecimento de Mãos", bg='#ADD8E6', font=titulo_font)
    titulo_label.pack(pady=20)

    # Imagem (ilustrativa)
    image_path = os.path.join('img', 'octocat.png')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        image = image.resize((200, 200), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(root, image=photo, bg='#ADD8E6')
        image_label.image = photo  # Mantém referência
        image_label.pack(pady=10)
    else:
        messagebox.showerror("Erro", f"Imagem não encontrada: {image_path}")

    # Botão Iniciar
    iniciar_button = tk.Button(root, text="Iniciar", command=iniciar, width=20, height=2)
    iniciar_button.pack(pady=10)

    # Botão Calibrar (treinar modelo)
    calibrar_button = tk.Button(root, text="Calibrar", command=calibrar, width=20, height=2)
    calibrar_button.pack(pady=10)

    # Barra de Progresso
    progress_var = tk.IntVar()
    progress_bar = tk.Scale(root, variable=progress_var, from_=0, to=100, orient=tk.HORIZONTAL, length=300, 
                            bg='#ADD8E6', troughcolor='white', fg='black', state='normal')
    progress_bar.pack(pady=20)

    progress_label = tk.Label(root, text="", bg='#ADD8E6', font=('Helvetica', 12))
    progress_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
