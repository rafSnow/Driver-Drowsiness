import os
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound  # Para alerta sonoro no Windows
import threading

class DetectorFadiga:
    def __init__(self):
        # Configurações adicionais para o MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_mesh_connections = mp.solutions.face_mesh_connections
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Pontos de referência dos olhos e boca
        self.p_olho_dir = [160, 144, 158, 153, 33, 133]
        self.p_olho_esq = [385, 380, 387, 373, 362, 263]
        self.p_boca = [13, 14]  # Ponto central do lábio superior e inferior

        # Configurações do modelo
        self.model_path = None  
        self.num_threads = 4    
        self.model_name = 'face_detection_full_face_landmarks.tflite'

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
        self.fps_list = []  
        self.MAX_FPS = 60   

        self.last_valid_landmarks = None
        self.landmark_smoothing = 0.6

        # Controle de cooldown para alertas sonoros
        self.last_beep_time = 0
        self.beep_cooldown = 1  # Segundos de intervalo mínimo entre beeps

        # ----------------------
        # NOVAS VARIÁVEIS (PERCLOS e LOG)
        # ----------------------
        # Manter variáveis do PERCLOS
        self.eye_closed_frames = []  # Armazena histórico de "olho fechado" (True/False)
        self.perclos_window = 150    # Quantidade de frames para cálculo PERCLOS
        # Explique essa quantidade de frames (150)? Por que 150? 
        
        self.perclos_threshold = 0.4 # Se > 40% do tempo com olhos fechados, indica fadiga
        self.perclos = 0.0           # Valor atual de PERCLOS

        # Novas variáveis para log periódico
        self.log_file_path = "fadiga_log.csv"  # Mudado para CSV
        self.log_interval = 60  # 1 minutos em segundos
        self.ultimo_log = time.time()
        self.piscadas_periodo = []
        self.ear_periodo = []
        self.mar_periodo = []
        self.eye_closure_durations = []
        # ----------------------

        # Update CSV header
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write("Timestamp,Piscadas/min,EAR_medio,MAR_medio,PERCLOS,Micro_cochilos,Bocejos,Tempo_Medio_Olhos_Fechados\n")


    def _suavizar_landmarks(self, current_landmarks):
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

    def _desenhar_pontos_principais(self, frame, face_landmarks):
        """Versão simplificada que desenha apenas os pontos essenciais"""
        PONTOS_ESSENCIAIS = {
            'Olho Dir': 263,
            'Olho Esq': 33,
            'Boca': 13
        }

        altura, largura = frame.shape[:2]
        for nome, idx in PONTOS_ESSENCIAIS.items():
            ponto = face_landmarks.landmark[idx]
            pos_x = int(ponto.x * largura)
            pos_y = int(ponto.y * altura)
            cv2.circle(frame, (pos_x, pos_y), 3, (0, 255, 0), -1)

    def _verificar_iluminacao(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brilho_medio = np.mean(gray)
        return brilho_medio > 50

    def _detectar_piscada(self, ear):
        EAR_PISCADA = 0.2 # para deixar mais sensível usar 0.2
        DURACAO_MAX = 0.25
        COOLDOWN = 0.08

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
                self.eye_closure_durations.append(duracao)  # Add duration to list
            self.piscando = False
            return True

        return False

    def _calculo_ear(self, face, p_olho_dir, p_olho_esq):
        face = np.array([[coord.x, coord.y] for coord in face])

        ear_esq = (np.linalg.norm(face[p_olho_esq[0]]-face[p_olho_esq[1]]) +
                   np.linalg.norm(face[p_olho_esq[2]]-face[p_olho_esq[3]])) / \
                  (2 * np.linalg.norm(face[p_olho_esq[4]]-face[p_olho_esq[5]]))

        ear_dir = (np.linalg.norm(face[p_olho_dir[0]]-face[p_olho_dir[1]]) +
                   np.linalg.norm(face[p_olho_dir[2]]-face[p_olho_dir[3]])) / \
                  (2 * np.linalg.norm(face[p_olho_dir[4]]-face[p_olho_dir[5]]))

        return (ear_esq + ear_dir) / 2

    def _calculo_mar(self, face, p_boca):
        face = np.array([[coord.x, coord.y] for coord in face])
        distancia_labios = np.abs(face[p_boca[0]][1] - face[p_boca[1]][1])
        return distancia_labios

    def _detectar_bocejo(self, mar):
        ABERTURA_THRESHOLD = 0.06
        DURACAO_MINIMA = 0.4
        COOLDOWN_PERIODO = 1.5

        tempo_atual = time.time()

        if tempo_atual - self.ultimo_bocejo < COOLDOWN_PERIODO:
            return False

        if mar > ABERTURA_THRESHOLD and not self.bocejando:
            self.t_inicial_bocejo = tempo_atual
            self.bocejando = True
            return False

        elif self.bocejando:
            duracao = tempo_atual - self.t_inicial_bocejo

            if mar < ABERTURA_THRESHOLD:
                if duracao >= DURACAO_MINIMA:
                    self.contagem_bocejos += 1
                    self.ultimo_bocejo = tempo_atual
                    self.bocejando = False
                    return True
                self.bocejando = False

        return False

    def _detectar_micro_cochilo(self, ear):
        EAR_THRESHOLD = 0.18
        MICRO_SLEEP_MIN = 1.5

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
        left_eye = np.mean([landmarks[p].y for p in self.p_olho_esq])
        right_eye = np.mean([landmarks[p].y for p in self.p_olho_dir])
        eye_level_diff = abs(left_eye - right_eye)
        face_rotation = abs(landmarks[self.p_olho_esq[0]].y - landmarks[self.p_olho_dir[0]].y)
        
        # Valores mais permissivos para rotação do rosto
        return eye_level_diff < 0.07 and face_rotation < 0.15

    # ----------------------
    # MÉTODO PARA ATUALIZAR PERCLOS
    # ----------------------
    def _atualizar_perclos(self, ear):
        EYE_CLOSE_THRESHOLD = 0.2
        olho_fechado = ear < EYE_CLOSE_THRESHOLD

        self.eye_closed_frames.append(olho_fechado)
        if len(self.eye_closed_frames) > self.perclos_window:
            self.eye_closed_frames.pop(0)

        # Cálculo de PERCLOS: % de frames com olhos fechados
        self.perclos = sum(self.eye_closed_frames) / len(self.eye_closed_frames)  # Store the value
        return self.perclos
    # ----------------------

    def _registrar_metricas_periodicas(self, ear, mar):
        """Registra métricas a cada intervalo definido"""
        tempo_atual = time.time()
        
        # Acumular dados do período
        self.ear_periodo.append(ear)
        self.mar_periodo.append(mar)
        if self.piscando:
            self.piscadas_periodo.append(1)
        
        # Verificar se é hora de registrar
        if tempo_atual - self.ultimo_log >= self.log_interval:
            minutos_decorridos = self.log_interval / 60  # Converter segundos para minutos
            piscadas_por_minuto = len(self.piscadas_periodo) / minutos_decorridos if minutos_decorridos > 0 else 0
            
            ear_medio = sum(self.ear_periodo) / len(self.ear_periodo) if self.ear_periodo else 0
            mar_medio = sum(self.mar_periodo) / len(self.mar_periodo) if self.mar_periodo else 0

            # Calculate average eye closure duration
            tempo_medio_fechado = sum(self.eye_closure_durations) / len(self.eye_closure_durations) if self.eye_closure_durations else 0
            
            # Formatar timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Registrar no arquivo CSV
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp},{piscadas_por_minuto:.2f},{ear_medio:.3f},"
                    f"{mar_medio:.3f},{self.perclos:.3f},{self.micro_cochilos},"
                    f"{self.contagem_bocejos},{tempo_medio_fechado:.3f}\n")
            
            # Resetar contadores do período
            self.piscadas_periodo = []
            self.ear_periodo = []
            self.mar_periodo = []
            self.ultimo_log = tempo_atual

    def _alerta_sonoro(self):
        """Executa o beep em uma thread separada."""
        winsound.Beep(1000, 200)

    def executar(self):
        cap = cv2.VideoCapture(0) 
        #resoluções sugeridas: 640x480, 1280x720, 1920x1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_frame_time = 0
        new_frame_time = 0

        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
        ) as face_mesh:

            while cap.isOpened():
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                self.fps_list.append(min(fps, self.MAX_FPS))
                if len(self.fps_list) > 30:
                    self.fps_list.pop(0)

                fps_medio = int(sum(self.fps_list) / len(self.fps_list))

                sucesso, frame = cap.read()
                if not sucesso:
                    continue

                iluminacao_adequada = self._verificar_iluminacao(frame)
                if not iluminacao_adequada:
                    cv2.putText(frame, "Iluminação inadequada", (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = face_mesh.process(frame_rgb)

                if resultados.multi_face_landmarks:
                    landmarks = resultados.multi_face_landmarks[0].landmark
                    if all(0.0 <= l.x <= 1.0 and 0.0 <= l.y <= 1.0 for l in landmarks):
                        if self._verificar_qualidade_deteccao(landmarks):
                            face_landmarks = self._suavizar_landmarks(resultados.multi_face_landmarks[0])

                            ear = self._calculo_ear(face_landmarks.landmark, self.p_olho_dir, self.p_olho_esq)
                            mar = self._calculo_mar(face_landmarks.landmark, self.p_boca)

                            # Registrar métricas periodicamente
                            self._registrar_metricas_periodicas(ear, mar)

                            bocejo = self._detectar_bocejo(mar)
                            micro_cochilo, tempo_fechado = self._detectar_micro_cochilo(ear)
                            piscada = self._detectar_piscada(ear)

                            # ----------------------
                            # ATUALIZA PERCLOS E VERIFICA ALERTA
                            # ----------------------
                            perclos = self._atualizar_perclos(ear)
                            tempo_atual = time.time()  # Defina o tempo antes
                            if perclos > self.perclos_threshold:
                                # Só disparar beep se já passou beep_cooldown
                                if tempo_atual - self.last_beep_time > self.beep_cooldown:
                                    self.last_beep_time = tempo_atual
                                    threading.Thread(target=self._alerta_sonoro).start()
                                cv2.putText(frame, "ALERTA: PERCLOS ALTO!", (20, 300),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            # ----------------------


                            # Mostrar métricas na tela
                            metricas = [
                                f"EAR: {ear:.2f}",
                                f"MAR: {mar:.2f}",
                                f"Bocejos: {self.contagem_bocejos}",
                                f"Micro-cochilos: {self.micro_cochilos}",
                                f"Piscadas: {self.contagem_piscadas}",
                                f"FPS: {fps_medio}",
                                f"PERCLOS: {perclos:.2f}"
                            ]

                            for i, metrica in enumerate(metricas):
                                cv2.putText(frame, metrica, (20, 30 * (i + 1)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            if micro_cochilo:
                                cv2.putText(frame, f"ALERTA: Olhos fechados por {tempo_fechado:.1f}s",
                                          (int(frame.shape[1]/2) - 200, int(frame.shape[0]/2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                                            (0, 0, 255), 3)

                            if bocejo:
                                cv2.putText(frame, "ALERTA: Bocejo detectado",
                                          (int(frame.shape[1]/2) - 150, int(frame.shape[0]/2)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                            self.mp_drawing.draw_landmarks(
                                frame,
                                face_landmarks,
                                self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                            )

                            self._desenhar_pontos_principais(frame, face_landmarks)

                        else:
                            cv2.putText(frame, "Rosto muito inclinado",
                                      (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Rosto fora do quadro",
                                  (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Nenhum rosto detectado",
                              (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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