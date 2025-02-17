import cv2
import dlib
import numpy as np

def calculate_ear(eye_points):
    """Calcula a razão de fechamento dos olhos (ECR)"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_points):
    """Calcula a razão de abertura da boca (MAR)"""
    A = np.linalg.norm(mouth_points[1] - mouth_points[7])
    B = np.linalg.norm(mouth_points[2] - mouth_points[6])
    C = np.linalg.norm(mouth_points[3] - mouth_points[5])
    D = np.linalg.norm(mouth_points[0] - mouth_points[4])
    mar = (A + B + C) / (3 * D)
    return mar

def detect_sleepiness():
    # Carregar detector e preditor de marcos faciais do Dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Captura de vídeo
    cap = cv2.VideoCapture(0)  # 0 para webcam padrão

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))

        # Converter para escala de cinza para melhor detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Extrair pontos dos olhos
            left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                        for i in range(36, 42)])
            right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                         for i in range(42, 48)])

            # Extrair pontos da boca
            mouth_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                     for i in range(48, 68)])

            # Calcular ECR para ambos os olhos
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            
            # Média dos ECRs
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(mouth_points)

            # Definir limiares
            ear_threshold = 0.20  # Ajustado para ser mais sensível
            mar_threshold = 0.4

            # Detectar sonolência
            if ear < ear_threshold or mar > mar_threshold:
                cv2.putText(frame, "ALERTA: Sonolência!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Desenhar marcos faciais de ambos os olhos
            for eye_points in [left_eye_points, right_eye_points]:
                for point in eye_points.astype(int):
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        cv2.imshow('Detecção de Sonolência', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_sleepiness()