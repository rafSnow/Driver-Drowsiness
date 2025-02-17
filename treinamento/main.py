import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0]-face_esq[1]) + np.linalg.norm(face_esq[2]-face_esq[3])) / (2 * np.linalg.norm(face_esq[4]-face_esq[5]))
        ear_dir = (np.linalg.norm(face_dir[0]-face_dir[1]) + np.linalg.norm(face_dir[2]-face_dir[3])) / (2 * np.linalg.norm(face_dir[4]-face_dir[5]))
    except:
        ear_esq, ear_dir = 0.0, 0.0

    return (ear_esq + ear_dir) / 2

def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]
        mar = (np.linalg.norm(face_boca[0]-face_boca[1]) + np.linalg.norm(face_boca[2]-face_boca[3]) + np.linalg.norm(face_boca[4]-face_boca[5])) / (2 * np.linalg.norm(face_boca[6] - face_boca[7]))
    except:
        mar = 0.0

    return mar

def desenhar_pontos(frame, face, pontos, cor, largura, comprimento):
    for id_coord in pontos:
        coord = mp_drawing._normalized_to_pixel_coordinates(face[id_coord].x, face[id_coord].y, largura, comprimento)
        if coord:
            cv2.circle(frame, coord, 2, cor, -1)

# Configuração do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Limiares
ear_limiar = 0.3
mar_limiar = 0.2
dormindo = 0
contagem_piscadas = 0
historico_piscadas = deque(maxlen=60)  # Últimos 60 segundos

t_inicial = time.time()
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print("Não foi possível capturar o frame")
            continue

        comprimento, largura, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        saida_facemesh = face_mesh.process(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if saida_facemesh.multi_face_landmarks:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1)
                )
                face = face_landmarks.landmark
                desenhar_pontos(frame, face, p_olhos, (255, 0, 0), largura, comprimento)
                desenhar_pontos(frame, face, p_boca, (0, 255, 0), largura, comprimento)

                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar = calculo_mar(face, p_boca)

                tempo_atual = time.time()
                if ear < ear_limiar and mar < mar_limiar:
                    if not dormindo:
                        t_inicial = tempo_atual
                        contagem_piscadas += 1
                        historico_piscadas.append(tempo_atual)
                        dormindo = 1
                else:
                    dormindo = 0

                while historico_piscadas and (tempo_atual - historico_piscadas[0]) > 60:
                    historico_piscadas.popleft()

                piscadas_pm = len(historico_piscadas)
                if piscadas_pm > 40:
                    status = "Alerta (Anormal)"
                    cor_status = (0, 0, 255)
                elif 20 <= piscadas_pm <= 40:
                    status = "Cansado"
                    cor_status = (0, 255, 255)
                elif 10 <= piscadas_pm < 20:
                    status = "Normal"
                    cor_status = (0, 255, 0)
                else:
                    status = "Sonolento"
                    cor_status = (0, 0, 255)
                tempo = (tempo_atual - t_inicial) if dormindo else 0.0

                cv2.rectangle(frame, (0,1), (290,140), (58,58,55), -1)
                cv2.putText(frame, f"EAR: {ear:.2f}", (1,24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (1,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame, f"Tempo: {tempo:.2f}", (1,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame, f"Piscadas/min: {piscadas_pm}", (1, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status}", (1, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_status, 2)

                if contagem_piscadas >= 10 and (piscadas_pm < 10 or tempo >= 1.5):
                    cv2.rectangle(frame, (0, 0), (largura, comprimento), (0, 0, 255), 5)  # Borda vermelha
                    cv2.putText(frame, "Alerta: Possível sonolência!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Detector de Sonolência", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
