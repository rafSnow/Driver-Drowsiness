import cv2
import mediapipe as mp
import numpy as np
import time


class DetectorFadiga:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pontos de referência dos olhos e boca
        self.p_olho_esq = [385, 380, 387, 373, 362, 263]
        self.p_olho_dir = [160, 144, 158, 153, 33, 133]
        # Atualizar pontos de referência da boca para usar apenas os lábios superior e inferior
        self.p_boca = [13, 14]  # Ponto central do lábio superior e inferior
        
        # Contadores e estados
        self.contagem_bocejos = 0
        self.micro_cochilos = 0
        self.ultimo_bocejo = 0
        self.bocejando = False
        self.dormindo = False
        self.contagem_piscadas = 0
        self.piscando = False
        self.ultimo_tempo_piscada = 0
        
        # Tempo inicial para estados
        self.t_inicial_olhos_fechados = 0
        self.t_inicial_bocejo = 0

        # Variáveis para cálculo de FPS
        self.fps_list = []  # Para calcular média móvel do FPS
        self.MAX_FPS = 30   # FPS máximo da câmera

        self.last_valid_landmarks = None
        self.landmark_smoothing = 0.7  # Fator de suavização dos landmarks

    def _suavizar_landmarks(self, current_landmarks): # Os landmarks são os pontos de referência do rosto
        """Suaviza os movimentos dos landmarks para reduzir tremulação"""
        if self.last_valid_landmarks is None:
            self.last_valid_landmarks = current_landmarks
            return current_landmarks
            
        smoothed = []
        for i, landmark in enumerate(current_landmarks.landmark):
            prev = self.last_valid_landmarks.landmark[i]
            smooth_x = prev.x * self.landmark_smoothing + landmark.x * (1 - self.landmark_smoothing)
            smooth_y = prev.y * self.landmark_smoothing + landmark.y * (1 - self.landmark_smoothing)
            smooth_z = prev.z * self.landmark_smoothing + landmark.z * (1 - self.landmark_smoothing)
            landmark.x, landmark.y, landmark.z = smooth_x, smooth_y, smooth_z
            
        self.last_valid_landmarks = current_landmarks
        return current_landmarks
    

    def _verificar_iluminacao(self, frame):
        """Verifica se a iluminação está adequada"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brilho_medio = np.mean(gray)
        return brilho_medio > 50  # Threshold de brilho mínimo

    def _detectar_piscada(self, ear):
        """Detecta piscadas"""
        EAR_PISCADA = 0.20      # Threshold para piscada
        DURACAO_MAX = 0.3       # Duração máxima de uma piscada normal
        COOLDOWN = 0.2          # Tempo mínimo entre piscadas
        
        tempo_atual = time.time()
        
        if tempo_atual - self.ultimo_tempo_piscada < COOLDOWN:
            return False
            
        if ear < EAR_PISCADA and not self.piscando:
            self.piscando = True
            self.ultimo_tempo_piscada = tempo_atual
            return False
            
        elif ear >= EAR_PISCADA and self.piscando:
            duracao = tempo_atual - self.ultimo_tempo_piscada
            if duracao < DURACAO_MAX:
                self.contagem_piscadas += 1
            self.piscando = False
            return True
            
        return False


    def _calculo_ear(self, face, p_olho_dir, p_olho_esq):
        """Calcula a razão de aspecto dos olhos (Eye Aspect Ratio)"""
        face = np.array([[coord.x, coord.y] for coord in face])
        
        # Calcular EAR para cada olho
        ear_esq = (np.linalg.norm(face[p_olho_esq[0]]-face[p_olho_esq[1]]) + 
                   np.linalg.norm(face[p_olho_esq[2]]-face[p_olho_esq[3]])) / \
                  (2 * np.linalg.norm(face[p_olho_esq[4]]-face[p_olho_esq[5]]))
        
        ear_dir = (np.linalg.norm(face[p_olho_dir[0]]-face[p_olho_dir[1]]) + 
                   np.linalg.norm(face[p_olho_dir[2]]-face[p_olho_dir[3]])) / \
                  (2 * np.linalg.norm(face[p_olho_dir[4]]-face[p_olho_dir[5]]))
        
        return (ear_esq + ear_dir) / 2

    def _calculo_mar(self, face, p_boca):
        """Calcula a distância vertical entre os lábios"""
        face = np.array([[coord.x, coord.y] for coord in face])
        # Calcula apenas a distância vertical entre os lábios
        distancia_labios = np.abs(face[p_boca[0]][1] - face[p_boca[1]][1])
        return distancia_labios

    def _detectar_bocejo(self, mar):
        """
        Detecta bocejos usando apenas a distância vertical entre os lábios
        """
        ABERTURA_THRESHOLD = 0.07      # Ajuste este valor conforme necessário
        DURACAO_MINIMA = 0.5          # Tempo mínimo que a boca deve ficar aberta
        COOLDOWN_PERIODO = 2.0        # Tempo mínimo entre bocejos
        
        tempo_atual = time.time()
        
        # Verificar período de cooldown
        if tempo_atual - self.ultimo_bocejo < COOLDOWN_PERIODO:
            return False
        
        # Detectar início do bocejo
        if mar > ABERTURA_THRESHOLD and not self.bocejando:
            self.t_inicial_bocejo = tempo_atual
            self.bocejando = True
            return False
        
        # Verificar se o bocejo completou a duração mínima
        elif self.bocejando:
            duracao = tempo_atual - self.t_inicial_bocejo
            
            if mar < ABERTURA_THRESHOLD:  # Boca fechou
                if duracao >= DURACAO_MINIMA:
                    self.contagem_bocejos += 1
                    self.ultimo_bocejo = tempo_atual
                    self.bocejando = False
                    return True
                self.bocejando = False
                
        return False

    def _detectar_micro_cochilo(self, ear):
        """Detecta micro-cochilos"""
        EAR_THRESHOLD = 0.19
        MICRO_SLEEP_MIN = 2.0
        
        if ear < EAR_THRESHOLD:
            if not self.dormindo:
                self.t_inicial_olhos_fechados = time.time()
                self.dormindo = True
                
            tempo_fechado = time.time() - self.t_inicial_olhos_fechados
            
            if tempo_fechado > MICRO_SLEEP_MIN:
                return True, tempo_fechado
        else:
            if self.dormindo:
                tempo_fechado = time.time() - self.t_inicial_olhos_fechados
                self.dormindo = False
                
                if tempo_fechado > MICRO_SLEEP_MIN:
                    self.micro_cochilos += 1
                    
        return False, 0.0

    def _verificar_qualidade_deteccao(self, landmarks):
        """Verifica se a detecção está com boa qualidade"""
        # Verificar simetria facial básica
        left_eye = np.mean([landmarks[p].y for p in self.p_olho_esq])
        right_eye = np.mean([landmarks[p].y for p in self.p_olho_dir])
        eye_level_diff = abs(left_eye - right_eye)
        
        # Verificar se o rosto está muito inclinado
        face_rotation = abs(landmarks[self.p_olho_esq[0]].y - landmarks[self.p_olho_dir[0]].y)
        
        return eye_level_diff < 0.05 and face_rotation < 0.1
    
    def _desenhar_texto_centralizado(self, frame, texto, cor=(0, 0, 255), escala=1.2):
        """Desenha texto centralizado na tela"""
        # Obter dimensões do frame
        altura, largura = frame.shape[:2]
        
        # Configurar fonte
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        espessura = 2
        
        # Obter dimensões do texto
        (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, escala, espessura)
        
        # Calcular posição central
        pos_x = (largura - largura_texto) // 2
        pos_y = (altura + altura_texto) // 2
        
        # Desenhar texto com contorno preto para melhor visibilidade
        cv2.putText(frame, texto, (pos_x, pos_y), fonte, escala, (0, 0, 0), espessura + 1)
        cv2.putText(frame, texto, (pos_x, pos_y), fonte, escala, cor, espessura)


    def executar(self):
        """Executa o detector de fadiga"""
        cap = cv2.VideoCapture(0)
        # Configurar resolução da câmera para melhor performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Variáveis para cálculo de FPS
        prev_frame_time = 0
        new_frame_time = 0
        
        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,           
                refine_landmarks=True,     
                min_detection_confidence=0.8,  
                min_tracking_confidence=0.8    
        ) as face_mesh:
            
            while cap.isOpened():
                # Cálculo do FPS
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                # Manter apenas os últimos 30 valores de FPS
                self.fps_list.append(min(fps, self.MAX_FPS))
                if len(self.fps_list) > 30:
                    self.fps_list.pop(0)
                
                # Calcular média móvel do FPS
                fps_medio = int(sum(self.fps_list) / len(self.fps_list))
                
                sucesso, frame = cap.read()
                if not sucesso:
                    continue

                # Verificar iluminação antes de processar o frame
                iluminacao_adequada = self._verificar_iluminacao(frame)
                if not iluminacao_adequada:
                    self._desenhar_texto_centralizado(frame, "Iluminação inadequada")
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = face_mesh.process(frame_rgb)
                
                if resultados.multi_face_landmarks:
                    # Verifica qualidade da detecção
                    landmarks = resultados.multi_face_landmarks[0].landmark
                    if all(0.0 <= l.x <= 1.0 and 0.0 <= l.y <= 1.0 for l in landmarks):
                        if self._verificar_qualidade_deteccao(landmarks):
                            face_landmarks = self._suavizar_landmarks(resultados.multi_face_landmarks[0])
                        
                            # Extrair métricas
                            ear = self._calculo_ear(face_landmarks.landmark, 
                                                self.p_olho_dir, self.p_olho_esq)
                            mar = self._calculo_mar(face_landmarks.landmark, self.p_boca)
                            
                            # Detectar eventos
                            bocejo = self._detectar_bocejo(mar)
                            micro_cochilo, tempo_fechado = self._detectar_micro_cochilo(ear)
                            piscada = self._detectar_piscada(ear)
                            
                            # Mostrar métricas na tela
                            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Bocejos: {self.contagem_bocejos}", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Micro-cochilos: {self.micro_cochilos}", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Piscadas: {self.contagem_piscadas}", (20, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"FPS: {fps_medio}", (20, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Alertas visuais
                            if micro_cochilo:
                                self._desenhar_texto_centralizado(
                                    frame, 
                                    f"ALERTA: Olhos fechados por {tempo_fechado:.1f}s", 
                                    cor=(0, 0, 255),
                                    escala=1.5
                                )
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                                            (0, 0, 255), 3)
                            
                            if bocejo:
                                self._desenhar_texto_centralizado(
                                    frame, 
                                    "ALERTA: Bocejo detectado", 
                                    cor=(0, 165, 255),
                                    escala=1.5
                                )
                            
                            # Desenhar landmarks faciais
                            self.mp_drawing.draw_landmarks(
                                frame,
                                face_landmarks,
                                self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                            )

                        else:
                            self._desenhar_texto_centralizado(
                                frame, 
                                "Rosto muito inclinado ou assimetrico"
                            )
                    else:
                        self._desenhar_texto_centralizado(
                            frame, 
                            "Rosto muito proximo ou fora do quadro"
                        )
                else:
                    self._desenhar_texto_centralizado(
                        frame, 
                        "Nenhum rosto detectado"
                    )
                
                cv2.imshow("Detector de Fadiga", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DetectorFadiga()
    try:
        detector.executar()
    except KeyboardInterrupt:
        print("\nDetector encerrado pelo usuário")