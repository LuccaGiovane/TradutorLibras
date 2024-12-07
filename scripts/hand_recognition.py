import cv2
import mediapipe as mp

def main():
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

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and results.multi_handedness:
                # Para cada mão, ajustaremos a posição do texto com base no índice i
                for i, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))

                    handedness = hand_info.classification[0].label  # "Left" ou "Right"

                    # Ajusta a posição vertical do texto (por exemplo, cada mão desce 40 pixels)
                    text_y = 30 + i * 40
                    cv2.putText(image, f"Mao Detectada: {handedness}",
                                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow("Reconhecimento de Mãos - Pressione 'q' para sair", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
