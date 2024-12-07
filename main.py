import os
import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk
import subprocess

def iniciar():
    # Executa o script hand_recognition.py para reconhecimento em tempo real das mãos
    subprocess.run(["python", "scripts/hand_recognition.py"])

def main():
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

    root.mainloop()

if __name__ == "__main__":
    main()
