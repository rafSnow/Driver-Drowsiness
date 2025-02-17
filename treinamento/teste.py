import cv2

cap = cv2.VideoCapture(0)  # Tenta abrir a primeira câmera
if not cap.isOpened():
    print("Erro ao acessar a câmera")
else:
    print("Câmera funcionando!")

ret, frame = cap.read()
if not ret:
    print("Falha ao capturar o vídeo. Tentando novamente...")

cap.release()
