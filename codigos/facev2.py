import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import logging
import argparse
import datetime
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detector_sonolencia.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("detector_sonolencia")

class DetectorSonolencia:
    def __init__(self, 
                 ear_limiar=0.3, 
                 mar_limiar=0.2,
                 uso_camera=0,
                 resolucao=(640, 480),
                 salvar_logs=False,
                 modo_debug=False):
        
        # Configurações
        self.ear_limiar = ear_limiar
        self.mar_limiar = mar_limiar
        self.uso_camera = uso_camera
        self.resolucao = resolucao
        self.salvar_logs = salvar_logs
        self.modo_debug = modo_debug
        
        # Estado
        self.dormindo = False
        self.contagem_piscadas = 0
        self.historico_piscadas = deque(maxlen=60)  # Últimos 60 segundos
        self.t_inicial = time.time()
        self.ultimo_alerta = 0
        
        # MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Pontos faciais
        self.p_olho_esq = [385, 380, 387, 373, 362, 263]
        self.p_olho_dir = [160, 144, 158, 153, 33, 133]
        self.p_olhos = self.p_olho_esq + self.p_olho_dir
        self.p_boca = [82, 87, 13, 14, 312, 317, 78, 308]
        
        # Preparar pasta para logs se necessário
        if self.salvar_logs:
            self.pasta_logs = os.path.join(
                os.getcwd(), 
                f"logs_sonolencia_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(self.pasta_logs, exist_ok=True)
            logger.info(f"Logs serão salvos em: {self.pasta_logs}")
    
    def calculo_ear(self, face, p_olho_dir, p_olho_esq):
        """Calcula a razão de aspecto dos olhos (Eye Aspect Ratio)"""
        try:
            face = np.array([[coord.x, coord.y] for coord in face])
            face_esq = face[p_olho_esq, :]
            face_dir = face[p_olho_dir, :]

            ear_esq = (np.linalg.norm(face_esq[0]-face_esq[1]) + 
                      np.linalg.norm(face_esq[2]-face_esq[3])) / (2 * np.linalg.norm(face_esq[4]-face_esq[5]))
            ear_dir = (np.linalg.norm(face_dir[0]-face_dir[1]) + 
                      np.linalg.norm(face_dir[2]-face_dir[3])) / (2 * np.linalg.norm(face_dir[4]-face_dir[5]))
            return (ear_esq + ear_dir) / 2
        except Exception as e:
            if self.modo_debug:
                logger.warning(f"Erro no cálculo de EAR: {str(e)}")
            return 0.0

    def calculo_mar(self, face, p_boca):
        """Calcula a razão de aspecto da boca (Mouth Aspect Ratio)"""
        try:
            face = np.array([[coord.x, coord.y] for coord in face])
            face_boca = face[p_boca, :]
            mar = (np.linalg.norm(face_boca[0]-face_boca[1]) + 
                  np.linalg.norm(face_boca[2]-face_boca[3]) + 
                  np.linalg.norm(face_boca[4]-face_boca[5])) / (2 * np.linalg.norm(face_boca[6] - face_boca[7]))
            return mar
        except Exception as e:
            if self.modo_debug:
                logger.warning(f"Erro no cálculo de MAR: {str(e)}")
            return 0.0

    def desenhar_pontos(self, frame, face, pontos, cor, largura, comprimento):
        """Desenha pontos de referência faciais no frame"""
        for id_coord in pontos:
            coord = self.mp_drawing._normalized_to_pixel_coordinates(
                face[id_coord].x, face[id_coord].y, 
                largura, comprimento
            )
            if coord:
                cv2.circle(frame, coord, 2, cor, -1)
    
    def avaliar_estado(self, piscadas_pm, tempo_olhos_fechados):
        """Determina o estado de atenção baseado nos parâmetros medidos"""
        if piscadas_pm > 40 or tempo_olhos_fechados >= 2.0:
            return "Alerta (Anormal)", (0, 0, 255)
        elif 20 <= piscadas_pm <= 40 or tempo_olhos_fechados >= 1.5:
            return "Cansado", (0, 255, 255)
        elif 10 <= piscadas_pm < 20:
            return "Normal", (0, 255, 0)
        else:
            return "Sonolento", (0, 0, 255)
    
    def atualizar_historico_piscadas(self, tempo_atual):
        """Mantém histórico de piscadas nos últimos 60 segundos"""
        while self.historico_piscadas and (tempo_atual - self.historico_piscadas[0]) > 60:
            self.historico_piscadas.popleft()
    
    def mostrar_informacoes(self, frame, ear, mar, tempo, piscadas_pm, status, cor_status):
        """Exibe informações na tela"""
        altura, largura, _ = frame.shape
        
        # Fundo semitransparente para o painel de informações
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (310, 150), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Informações
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tempo olhos fechados: {tempo:.2f}s", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Piscadas/min: {piscadas_pm}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
        
        # Mostrar alerta se necessário
        if status.startswith("Alerta") and time.time() - self.ultimo_alerta > 3:
            # Piscar borda de alerta
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (largura, altura), (0, 0, 255), 5)
                cv2.putText(frame, "ALERTA: SONOLÊNCIA DETECTADA!", 
                          (largura//2 - 220, altura//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            if time.time() - self.ultimo_alerta > 10:  # Salvar log de alerta a cada 10s
                self.ultimo_alerta = time.time()
                logger.warning(f"Alerta de sonolência: EAR={ear:.2f}, Tempo={tempo:.2f}s")
                
                if self.salvar_logs:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(os.path.join(self.pasta_logs, f"alerta_{timestamp}.jpg"), frame)
    
    def executar(self):
        """Inicia o detector de sonolência"""
        cap = cv2.VideoCapture(self.uso_camera)
        
        # Configurar resolução
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolucao[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolucao[1])
        
        if not cap.isOpened():
            logger.error("Erro ao abrir câmera")
            return
        
        logger.info("Iniciando detector de sonolência...")
        
        with self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while cap.isOpened():
                sucesso, frame = cap.read()
                if not sucesso:
                    logger.warning("Frame não capturado")
                    # Tentar novamente por alguns frames antes de desistir
                    tentativas = 0
                    while not sucesso and tentativas < 5:
                        sucesso, frame = cap.read()
                        tentativas += 1
                    if not sucesso:
                        logger.error("Não foi possível recuperar a conexão com a câmera")
                        break
                
                altura, largura, _ = frame.shape
                
                # Processar imagem
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                saida_facemesh = face_mesh.process(frame_rgb)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                tempo_atual = time.time()
                
                if saida_facemesh.multi_face_landmarks:
                    for face_landmarks in saida_facemesh.multi_face_landmarks:
                        # Desenhar malha facial se em modo debug
                        if self.modo_debug:
                            self.mp_drawing.draw_landmarks(
                                frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                                    color=(255, 102, 102), thickness=1, circle_radius=1),
                                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                    color=(102, 204, 0), thickness=1, circle_radius=1)
                            )
                        
                        face = face_landmarks.landmark
                        self.desenhar_pontos(frame, face, self.p_olhos, (255, 0, 0), largura, altura)
                        self.desenhar_pontos(frame, face, self.p_boca, (0, 255, 0), largura, altura)
                        
                        # Calcular métricas
                        ear = self.calculo_ear(face, self.p_olho_dir, self.p_olho_esq)
                        mar = self.calculo_mar(face, self.p_boca)
                        
                        # Verificar estado de sonolência
                        if ear < self.ear_limiar and mar < self.mar_limiar:
                            if not self.dormindo:
                                self.t_inicial = tempo_atual
                                self.contagem_piscadas += 1
                                self.historico_piscadas.append(tempo_atual)
                                self.dormindo = True
                        else:
                            self.dormindo = False
                        
                        # Atualizar histórico de piscadas
                        self.atualizar_historico_piscadas(tempo_atual)
                        
                        # Calcular métricas
                        piscadas_pm = len(self.historico_piscadas)
                        tempo_olhos_fechados = (tempo_atual - self.t_inicial) if self.dormindo else 0.0
                        
                        # Avaliar estado
                        status, cor_status = self.avaliar_estado(piscadas_pm, tempo_olhos_fechados)
                        
                        # Mostrar informações
                        self.mostrar_informacoes(frame, ear, mar, tempo_olhos_fechados, 
                                              piscadas_pm, status, cor_status)
                else:
                    # Nenhum rosto detectado
                    cv2.putText(frame, "Nenhum rosto detectado", (largura//4, altura//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Detector de Sonolência", frame)
                
                # Permitir saída com 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Limpeza
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Detector de sonolência encerrado")


def main():
    # Parse argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Detector de Sonolência")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão: 0)")
    parser.add_argument("--ear", type=float, default=0.3, help="Limiar de EAR (padrão: 0.3)")
    parser.add_argument("--mar", type=float, default=0.2, help="Limiar de MAR (padrão: 0.2)")
    parser.add_argument("--debug", action="store_true", help="Ativa modo de depuração")
    parser.add_argument("--logs", action="store_true", help="Salva logs de alertas")
    parser.add_argument("--resolucao", type=str, default="640x480", 
                      help="Resolução da câmera (padrão: 640x480)")
    
    args = parser.parse_args()
    
    # Converter string de resolução para tupla
    largura, altura = map(int, args.resolucao.split('x'))
    
    # Iniciar detector
    detector = DetectorSonolencia(
        ear_limiar=args.ear,
        mar_limiar=args.mar,
        uso_camera=args.camera,
        resolucao=(largura, altura),
        salvar_logs=args.logs,
        modo_debug=args.debug
    )
    
    detector.executar()


if __name__ == "__main__":
    main()