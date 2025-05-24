import cv2
import mediapipe as mp
import serial
import time

# Conexão com o Arduino (ajuste a porta conforme necessário)
arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Captura de vídeo
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotaciona o frame 90 graus no sentido horário
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Cabeça: usando o nariz (índice 0)
        head = landmarks[0]
        head_coords = (int(head.x * w), int(head.y * h))
        cv2.putText(frame, f"Head: {head_coords}", (head_coords[0], head_coords[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Mão esquerda: usando o pulso esquerdo (índice 15)
        left_hand = landmarks[15]
        left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
        cv2.putText(frame, f"Head: {left_hand_coords}", (left_hand_coords[0], left_hand_coords[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Mão direita: usando o pulso direito (índice 16)
        right_hand = landmarks[16]
        right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
        cv2.putText(frame, f"Head: {right_hand_coords}", (right_hand_coords[0], right_hand_coords[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # pálpebra superior e inferior do olho esquerdo
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        # Converte para coordenadas reais
        left_eye_top_y = int(left_eye_top.y * h)
        left_eye_bottom_y = int(left_eye_bottom.y * h)

        # pálpebra superior e inferior do olho direito
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        # Converte para coordenadas reais
        right_eye_top_y = int(right_eye_top.y * h)
        right_eye_bottom_y = int(right_eye_bottom.y * h)

        # Calcula a distância vertical dos olhos
        left_eye_diff = abs(left_eye_top_y - left_eye_bottom_y)
        right_eye_diff = abs(right_eye_top_y - right_eye_bottom_y)

        threshold = 4  # pixels

        # Verificações (baseadas no eixo X agora)
        comando = ""
        # estamos usando o [1] para se referir ao eixo y
        if left_hand_coords[1] < head_coords[1]:  # Mão esquerda acima da cabeça
            comando += "G"  # LED verde
        if right_hand_coords[1] < head_coords[1]:  # Mão direita acima da cabeça
            comando += "R"  # LED vermelho
        if left_eye_diff < threshold:  # quando o olho esquerdo estiver fechado
            comando += "Y"  # LED amarelo
        if right_eye_diff < threshold:  # quando o olho direito estiver fechado
            comando += "B"  # LED azul

        if comando:
            arduino.write(comando.encode())  # Envia o comando para o Arduino

    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
