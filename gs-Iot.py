import mediapipe as mp
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
# modelo YOLO para detectar pessoas

cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()
# abre o vídeo

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)
# configuração do midiapipe

skip_frames = 3 
# Fator de aceleração: pular n frames

while cap.isOpened():
    for _ in range(skip_frames):
        cap.read()  # Pula frames

    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, None, fx=1, fy=1)
    results = model(frame_resized)
    detections = results[0]
    boxes = detections.boxes.xyxy
    classes = detections.boxes.cls

    # Filtra apenas pessoas (classe 0)
    person_boxes = [box for box, cls in zip(boxes, classes) if int(cls) == 0]
    #print(person_boxes)

    # Cria uma cópia do frame para desenhar só as pessoas
    annotated_frame = frame_resized.copy()

    person_crops = []

    for box in person_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2) # Desenha um retângulo verde só nas pessoas
        crop = frame[y_min:y_max, x_min:x_max] 
        person_crops.append(crop)
    #print(person_crops)

    for idx, crop in enumerate(person_crops):
        # Passo 1: Converte o crop para RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Passo 2: Processa o recorte com MediaPipe
        results = pose.process(crop_rgb)
        
        # Passo 3: Checa se encontrou landmarks
        """if results.pose_landmarks:
            print(f"Pessoa {idx}: Landmarks detectados")
        else:
            print(f"Pessoa {idx}: Nenhum landmark detectado")"""

    display_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
    cv2.imshow('YOLO Detection', display_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

