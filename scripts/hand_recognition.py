import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

def main():
    # Carrega o modelo treinado
    model_path = "scripts/model.h5"
    model = tf.keras.models.load_model(model_path)

    # Classes usadas no treinamento (mesma ordem utilizada no treinamento)
    valid_letters = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in ['H','J','K','X','Z'] and letter <= 'Y']
    # Ex: Se no treinamento as classes foram definidas assim, elas estarão em ordem alfabética
    # valid_letters = ['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W','Y'] 
    # Certifique-se de que "valid_letters" bate com o "class_names" do treinamento.

    img_height = 64
    img_width = 64

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Não foi possível acessar a câmera.")
            return

        print("Pressione 'q' para sair.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Converte para RGB pois o mediapipe usa RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Se mãos detectadas
            if results.multi_hand_landmarks and results.multi_handedness:
                for i, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

                    handedness = hand_info.classification[0].label  # "Left" ou "Right"

                    # Obter os pontos (landmarks) da mão
                    h, w, c = image.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                    # Calcula a bounding box da mão
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Opcional: aumentar um pouco a caixa ao redor da mão
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Extrai a imagem da mão
                    hand_img = image[y_min:y_max, x_min:x_max]

                    # Verifica se a extração foi bem sucedida
                    if hand_img.size != 0:
                        # Redimensiona para o tamanho esperado pelo modelo
                        hand_img = cv2.resize(hand_img, (img_width, img_height))
                        # Normaliza (mesma normalização do treinamento: layers.Rescaling(1./255))
                        hand_img = hand_img.astype('float32') / 255.0
                        # Adiciona dimensão batch
                        hand_img = np.expand_dims(hand_img, axis=0)

                        # Faz a previsão
                        predictions = model.predict(hand_img)
                        class_idx = np.argmax(predictions, axis=1)[0]
                        predicted_letter = valid_letters[class_idx]

                        # Desenha a previsão na tela
                        text_y = 30 + i * 40
                        cv2.putText(image, f"Letra: {predicted_letter} ({handedness})",
                                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    else:
                        # Caso não seja possível extrair corretamente a imagem da mão
                        text_y = 30 + i * 40
                        cv2.putText(image, f"Mao Detectada: {handedness} - (Sem ROI)",
                                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Reconhecimento de Mãos - Pressione 'q' para sair", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
