import os
import threading
import tkinter as tk
from tkinter import messagebox, font
from tkinter import ttk
from PIL import Image, ImageTk

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Configurações gerais
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Letras válidas (as mesmas do treinamento)
VALID_LETTERS = [
    letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if letter not in ['H','J','K','X','Z'] and letter <= 'Y'
]


# ---------------------- Funções de treinamento ----------------------
def train_model(progress_var, progress_bar, progress_label, calibrar_button, root):
    """
    Função executada em thread separada para treinar o modelo
    """
    calibrar_button.config(state='disabled')
    progress_var.set(0)
    progress_bar['value'] = 0
    progress_label.config(text="Treinando... 0%")
    
    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    epochs = 20
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        labels='inferred',
        class_names=VALID_LETTERS
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        labels='inferred',
        class_names=VALID_LETTERS
    )

    AUTOTUNE = tf.data.AUTOTUNE
    val_size = 0.2
    train_size = int((1 - val_size) * tf.data.experimental.cardinality(train_ds).numpy())
    val_ds = train_ds.skip(train_size)
    train_ds = train_ds.take(train_size)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(VALID_LETTERS)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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

    class ProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progresso = int(((epoch+1) / epochs) * 100)
            progress_var.set(progresso)
            progress_bar['value'] = progresso
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

    progress_var.set(100)
    progress_bar['value'] = 100
    progress_label.config(text="Treinamento concluído!")
    calibrar_button.config(state='normal')


def calibrar(progress_var, progress_bar, progress_label, calibrar_button, root):
    """
    Função chamada pelo botão Calibrar. Cria a thread de treinamento.
    """
    progress_bar.pack(pady=10)
    progress_label.pack(pady=5)

    train_thread = threading.Thread(
        target=train_model,
        args=(progress_var, progress_bar, progress_label, calibrar_button, root),
        daemon=True
    )
    train_thread.start()


# ---------------------- Classe Principal da Aplicação ----------------------
class SignTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento de Mãos")
        self.root.configure(bg='#ADD8E6')
        self.root.geometry('600x750')
        self.root.resizable(False, False)

        # Fonte para o título
        titulo_font = font.Font(family='Helvetica', size=24, weight='bold')
        titulo_label = tk.Label(self.root, text="Reconhecimento de Mãos",
                                bg='#ADD8E6', font=titulo_font)
        titulo_label.pack(pady=10)

        # Frame para agrupar os botões
        btn_frame = tk.Frame(self.root, bg='#ADD8E6')
        btn_frame.pack(pady=5)

        # Botão para iniciar
        self.iniciar_button = tk.Button(btn_frame,
                                        text="Iniciar Reconhecimento",
                                        width=18, height=2,
                                        command=self.iniciar_reconhecimento)
        self.iniciar_button.grid(row=0, column=0, padx=5, pady=5)

        # Botão para parar
        self.parar_button = tk.Button(btn_frame,
                                      text="Parar Reconhecimento",
                                      width=18, height=2,
                                      command=self.parar_reconhecimento)
        self.parar_button.grid(row=0, column=1, padx=5, pady=5)

        # Botão Calibrar
        self.calibrar_button = tk.Button(btn_frame,
                                         text="Calibrar Modelo",
                                         width=18, height=2,
                                         command=self.calibrar_modelo)
        self.calibrar_button.grid(row=0, column=2, padx=5, pady=5)

        # Barra de Progresso e Label de Progresso (inicialmente ocultos)
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.root,
                                            variable=self.progress_var,
                                            maximum=100, length=300,
                                            mode='determinate')
        self.progress_label = tk.Label(self.root,
                                       text="",
                                       bg='#ADD8E6',
                                       font=('Helvetica', 12))

        # Label para exibir o vídeo
        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(pady=10)

        # "Console de saída": caixa de texto para exibir as letras
        lbl_chat = tk.Label(self.root, text="Letras Reconhecidas:",
                            bg='#ADD8E6', font=('Helvetica', 14, 'bold'))
        lbl_chat.pack(pady=5)

        self.letter_console = tk.Text(self.root, height=6,
                                      width=60, font=("Helvetica", 14))
        self.letter_console.pack(pady=5)

        # Para exibir a imagem (octocat) abaixo da caixa de texto, se você quiser
        image_path = os.path.join('img', 'octocat.png')
        if os.path.exists(image_path):
            from PIL import Image
            octo_img = Image.open(image_path)
            octo_img = octo_img.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(octo_img)
            image_label = tk.Label(self.root, image=photo, bg='#ADD8E6')
            image_label.image = photo
            image_label.pack(pady=10)
        else:
            # Opcional: messagebox.showerror("Erro", f"Imagem não encontrada: {image_path}")
            pass

        # Flag de captura e thread
        self.capturing = False
        self.cap_thread = None

        # Modelo e MediaPipe
        self.model = None
        self.mp_hands = None
        self.mp_drawing = None

    def calibrar_modelo(self):
        calibrar(self.progress_var, self.progress_bar,
                 self.progress_label, self.calibrar_button, self.root)

    def iniciar_reconhecimento(self):
        """
        Inicia a captura de vídeo e o loop de reconhecimento em uma thread.
        """
        if self.capturing:
            messagebox.showinfo("Info", "O reconhecimento já está em execução.")
            return

        model_path = "scripts/model.h5"  # Ajustar se necessário
        if not os.path.exists(model_path):
            messagebox.showerror("Erro",
                                 f"Arquivo de modelo não encontrado em {model_path}.\nTreine ou mova o .h5 corretamente.")
            return

        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar o modelo: {e}")
            return

        # Carrega MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.capturing = True
        self.cap_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.cap_thread.start()

    def parar_reconhecimento(self):
        """
        Para o loop de captura caso esteja em execução
        """
        if self.capturing:
            self.capturing = False
            # Espera a thread encerrar
            if self.cap_thread is not None:
                self.cap_thread.join()
            messagebox.showinfo("Info", "Reconhecimento Parado.")
        else:
            messagebox.showinfo("Info", "O reconhecimento não está em execução.")

    def capture_loop(self):
        """
        Loop de captura de vídeo e reconhecimento, executado em thread separada
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
            self.capturing = False
            return

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            while self.capturing:
                ret, frame = cap.read()
                if not ret:
                    break

                # Converte para RGB para o MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # Volta para BGR para desenhar
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Se mãos detectadas
                if results.multi_hand_landmarks and results.multi_handedness:
                    for i, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks,
                                                                       results.multi_handedness)):
                        self.mp_drawing.draw_landmarks(
                            image_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )

                        handedness = hand_info.classification[0].label  # "Left" ou "Right"
                        h, w, c = image_bgr.shape
                        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))

                        padding = 20
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(w, x_max + padding)
                        y_max = min(h, y_max + padding)

                        hand_img = image_bgr[y_min:y_max, x_min:x_max]

                        if hand_img.size != 0:
                            # Redimensiona e normaliza
                            hand_img_resized = cv2.resize(hand_img, (IMG_WIDTH, IMG_HEIGHT))
                            hand_img_resized = hand_img_resized.astype('float32') / 255.0
                            hand_img_resized = np.expand_dims(hand_img_resized, axis=0)

                            predictions = self.model.predict(hand_img_resized)
                            class_idx = np.argmax(predictions, axis=1)[0]
                            predicted_letter = VALID_LETTERS[class_idx]

                            # Desenha a previsão na tela
                            text_y = 30 + i * 40
                            cv2.putText(image_bgr,
                                        f"Letra: {predicted_letter} ({handedness})",
                                        (10, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # Atualiza o "chat" de letras
                            self.root.after(0, self.append_letter_to_console, predicted_letter)

                # Converter para exibir no Tkinter
                img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img_pil)

                # Atualiza o label de vídeo
                self.root.after(0, self.update_video_label, imgtk)

                # Verifica tecla 'q'
                # (Funciona melhor quando a janela OpenCV está em foco, mas mantemos como referência)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.capturing = False

        cap.release()
        cv2.destroyAllWindows()
        self.capturing = False

    def update_video_label(self, imgtk):
        """
        Atualiza a Label de vídeo com a nova imagem.
        """
        self.video_label.config(image=imgtk)
        self.video_label.imgtk = imgtk

    def append_letter_to_console(self, letter):
        """
        Insere a letra reconhecida na caixa de texto.
        """
        self.letter_console.insert(tk.END, letter)


def main():
    root = tk.Tk()
    app = SignTranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
