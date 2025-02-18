import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import logging
import os
import argparse
import json
import threading
import pygame
import requests
from collections import deque
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib


class DetectorFadigaAvancado:
    def __init__(self, config_file=None):
        # Carregar configuração ou usar padrões
        self.config = self._carregar_config(config_file)
        self.ultimo_bocejo = 0

        # Inicializar subsistemas
        self._inicializar_logging()
        self._inicializar_alertas()
        self._inicializar_mediapipe()
        self._inicializar_modelos_ml()
        self._inicializar_sessao()

        # Iniciar threads de monitoramento secundário
        self._iniciar_threads_secundarias()

    def _carregar_config(self, config_file):
        """Carrega configuração de um arquivo JSON ou usa valores padrão"""
        config_padrao = {
            # Configurações gerais
            "camera_index": 0,
            "resolucao": (640, 480),
            "modo_debug": False,
            "salvar_logs": True,

            # Limiares de detecção
            "ear_limiar": 0.25,
            "mar_limiar": 0.2,
            "head_movement_threshold": 0.2,
            "microsleep_duration_threshold": 1.5,

            # Configurações de alerta
            "alerta_visual": True,
            "alerta_sonoro": True,
            "alerta_vibracao": False,
            "intervalo_min_alertas": 3,  # segundos
            "escalate_after_alerts": 3,

            # Configurações de aprendizado
            "usar_modelo_personalizado": False,
            "coletar_dados_treinamento": True,
            "intervalo_retreinamento": 7,  # dias

            # Características avançadas
            "analise_pupila": True,
            "deteccao_bocejos": True,
            "analise_postura": True,
            "monitorar_volante": False,  # requer hardware

            # Caminhos
            "pasta_logs": "logs",
            "pasta_modelos": "models",
            "pasta_dados": "data",
            "som_alerta": "sounds/alert.mp3"
        }

        # Tentar carregar configuração de arquivo
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_usuario = json.load(f)
                    # Mesclar configurações, mantendo padrões para valores ausentes
                    for k, v in config_usuario.items():
                        config_padrao[k] = v
                self.logger.info(f"Configuração carregada de {config_file}")
            except Exception as e:
                self.logger.error(f"Erro ao carregar configuração: {e}")

        return config_padrao

    def _inicializar_logging(self):
        """Configura sistema de logging"""
        # Criar diretórios se não existirem
        for pasta in [self.config["pasta_logs"], self.config["pasta_modelos"], self.config["pasta_dados"]]:
            os.makedirs(pasta, exist_ok=True)

        # Nome do arquivo de log com timestamp
        log_file = os.path.join(
            self.config["pasta_logs"],
            f"sessao_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Configurar logging
        logging.basicConfig(
            level=logging.DEBUG if self.config["modo_debug"] else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("detector_fadiga")
        self.logger.info("Sistema de logging inicializado")

    def _inicializar_alertas(self):
        """Inicializa sistema de alertas sonoros e visuais"""
        self.ultimo_alerta = 0
        self.contador_alertas = 0

        # Inicializar pygame para alertas sonoros
        if self.config["alerta_sonoro"]:
            try:
                pygame.mixer.init()
                self.som_alerta = pygame.mixer.Sound(self.config["som_alerta"])
                self.logger.info("Sistema de alertas sonoros inicializado")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar alertas sonoros: {e}")
                self.config["alerta_sonoro"] = False

    def _inicializar_mediapipe(self):
        """Inicializa componentes do MediaPipe para detecção facial"""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

        # Pontos de referência faciais
        self.p_olho_esq = [385, 380, 387, 373, 362, 263]
        self.p_olho_dir = [160, 144, 158, 153, 33, 133]
        self.p_olhos = self.p_olho_esq + self.p_olho_dir
        self.p_boca = [
            0, 17,    # Top outer
            37, 267,  # Top inner
            269, 405,  # Bottom outer
            291, 314  # Bottom inner
        ]
        self.p_pupila_esq = [468, 469, 470, 471, 472]
        self.p_pupila_dir = [473, 474, 475, 476, 477]
        # pontos para orientação da cabeça
        self.p_cabeca = [8, 33, 263, 61, 291, 199]

        # Histórico para análise de tendências
        # 2 minutos de histórico em 30 FPS
        self.ear_historico = deque(maxlen=120)
        self.mar_historico = deque(maxlen=120)
        self.head_pose_historico = deque(maxlen=120)
        self.historico_piscadas = deque(maxlen=180)  # 3 minutos

        # Parâmetros de estado
        self.dormindo = False
        self.bocejando = False
        self.cabeca_inclinada = False
        self.t_inicial_olhos_fechados = time.time()
        self.t_inicial_bocejo = time.time()
        self.t_inicial_cabeca = time.time()
        self.micro_cochilos = 0
        self.contagem_piscadas = 0
        self.contagem_bocejos = 0

        self.logger.info("MediaPipe inicializado")

    def _inicializar_modelos_ml(self):
        """Inicializa ou carrega modelos de machine learning"""
        modelo_path = os.path.join(
            self.config["pasta_modelos"], "anomaly_detector.joblib")

        # Tentar carregar modelo existente ou criar novo
        if os.path.exists(modelo_path) and self.config["usar_modelo_personalizado"]:
            try:
                self.anomaly_detector = joblib.load(modelo_path)
                self.logger.info(
                    f"Modelo de detecção de anomalias carregado de {modelo_path}")
            except Exception as e:
                self.logger.warning(
                    f"Erro ao carregar modelo: {e}. Criando novo modelo.")
                self.anomaly_detector = IsolationForest(
                    contamination=0.05, random_state=42)
        else:
            self.anomaly_detector = IsolationForest(
                contamination=0.05, random_state=42)
            self.logger.info("Novo modelo de detecção de anomalias criado")

        # Histórico de dados para treinamento
        self.dados_treinamento = []

    def _inicializar_sessao(self):
        """Inicializa parâmetros da sessão atual"""
        self.inicio_sessao = time.time()
        self.dados_sessao = {
            "timestamp_inicio": datetime.datetime.now().isoformat(),
            "alertas": [],
            "metricas": {
                "piscadas_total": 0,
                "bocejos_total": 0,
                "micro_cochilos": 0,
                "duracao_total": 0,
                "nivel_fadiga_medio": 0,
                "alertas_total": 0
            }
        }
        self.cap = None  # Será inicializado em executar()

    def _iniciar_threads_secundarias(self):
        """Inicia threads para monitoramento secundário e processamento assíncrono"""
        # Thread para análise de tendências de longo prazo
        self.stop_threads = False
        self.thread_analise = threading.Thread(
            target=self._analisar_tendencias_thread)
        self.thread_analise.daemon = True
        self.thread_analise.start()

        # Thread para salvar dados periodicamente
        self.thread_salvar = threading.Thread(
            target=self._salvar_dados_periodicamente)
        self.thread_salvar.daemon = True
        self.thread_salvar.start()

        self.logger.info("Threads secundárias iniciadas")

    def _calculo_ear(self, face, p_olho_dir, p_olho_esq):
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
            if self.config["modo_debug"]:
                self.logger.warning(f"Erro no cálculo de EAR: {str(e)}")
            return 0.0

    def _calculo_mar(self, face, p_boca):
        """Calcula a razão de aspecto da boca (Mouth Aspect Ratio)"""
        try:
            face = np.array([[coord.x, coord.y] for coord in face])
            face_boca = face[p_boca, :]
            mar = (np.linalg.norm(face_boca[0]-face_boca[1]) +
                   np.linalg.norm(face_boca[2]-face_boca[3]) +
                   np.linalg.norm(face_boca[4]-face_boca[5])) / (2 * np.linalg.norm(face_boca[6] - face_boca[7]))
            return mar
        except Exception as e:
            if self.config["modo_debug"]:
                self.logger.warning(f"Erro no cálculo de MAR: {str(e)}")
            return 0.0

    def _calculo_diametro_pupila(self, face, p_pupila):
        """Calcula o diâmetro aproximado da pupila"""
        try:
            face = np.array([[coord.x, coord.y] for coord in face])
            pupila = face[p_pupila, :]
            centro = np.mean(pupila, axis=0)
            raios = [np.linalg.norm(p - centro) for p in pupila]
            diametro = np.mean(raios) * 2
            return diametro
        except Exception as e:
            if self.config["modo_debug"]:
                self.logger.warning(
                    f"Erro no cálculo do diâmetro da pupila: {str(e)}")
            return 0.0

    def _calculo_orientacao_cabeca(self, face, p_cabeca):
        """Estima a orientação da cabeça a partir de pontos de referência"""
        try:
            face = np.array([[coord.x, coord.y, coord.z] for coord in face])
            pontos_cabeca = face[p_cabeca, :]

            # Calcular vetores direcionais
            vetor_frontal = pontos_cabeca[0] - pontos_cabeca[3]  # frente-trás
            vetor_lateral = pontos_cabeca[1] - \
                pontos_cabeca[2]  # direita-esquerda
            vetor_vertical = pontos_cabeca[4] - pontos_cabeca[5]  # cima-baixo

            # Normalizar vetores
            vetor_frontal = vetor_frontal / np.linalg.norm(vetor_frontal)
            vetor_lateral = vetor_lateral / np.linalg.norm(vetor_lateral)
            vetor_vertical = vetor_vertical / np.linalg.norm(vetor_vertical)

            # Retornar componentes de orientação
            return vetor_frontal, vetor_lateral, vetor_vertical
        except Exception as e:
            if self.config["modo_debug"]:
                self.logger.warning(
                    f"Erro no cálculo da orientação da cabeça: {str(e)}")
            return np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])

    def _detectar_bocejo(self, mar, ear):
        """Detecta bocejos com base em MAR e outros indicadores"""
        # Aumentar precisão com limiares mais restritivos
        MAR_THRESHOLD = 0.75          # Reduzido de 0.85
        MAR_MIN_THRESHOLD = 0.25      # Aumentado de 0.2
        DURACAO_MINIMA = 1.5          # Reduzido de 2.0
        DURACAO_MAXIMA = 4.0          # Reduzido de 6.0
        COOLDOWN_PERIODO = 8.0        # Aumentado de 5.0

        try:
            tempo_atual = time.time()

            # Verificar cooldown
            if hasattr(self, 'ultimo_bocejo') and \
                    (tempo_atual - self.ultimo_bocejo) < COOLDOWN_PERIODO:
                return False

            # Adicionar verificação de histórico de MAR
            mar_medio = np.mean(list(self.mar_historico)
                                [-10:]) if self.mar_historico else mar
            mar_std = np.std(list(self.mar_historico)
                             [-10:]) if self.mar_historico else 0

            # Verificar se o MAR atual é significativamente maior que a média recente
            if mar < MAR_MIN_THRESHOLD or mar < (mar_medio + 2*mar_std):
                self.bocejando = False
                return False

            if mar > MAR_THRESHOLD:
                if not self.bocejando:
                    self.t_inicial_bocejo = tempo_atual
                    self.bocejando = True
                    if self.config["modo_debug"]:
                        self.logger.debug(
                            f"Possível bocejo iniciado - MAR: {mar:.2f}")

                duracao = tempo_atual - self.t_inicial_bocejo

                if duracao > DURACAO_MAXIMA:
                    self.bocejando = False
                    return False

                # Verificar padrão completo do bocejo
                if DURACAO_MINIMA < duracao < DURACAO_MAXIMA and self.bocejando:
                    # Verificar se houve uma progressão natural do MAR
                    # assumindo 30 FPS
                    mar_seq = list(self.mar_historico)[-int(duracao*30):]
                    if len(mar_seq) > 5:
                        # Verificar se houve aumento e depois diminuição do MAR
                        max_idx = np.argmax(mar_seq)
                        # pico não está nas extremidades
                        if 0 < max_idx < len(mar_seq)-1:
                            self.contagem_bocejos += 1
                            self.bocejando = False
                            self.ultimo_bocejo = tempo_atual
                            self.logger.info(
                                f"Bocejo detectado - MAR: {mar:.2f}, Duração: {duracao:.2f}s")
                            return True
            else:
                if self.bocejando:
                    self.logger.debug(f"Bocejo cancelado - MAR: {mar:.2f}")
                self.bocejando = False

            return False

        except Exception as e:
            self.logger.error(f"Erro na detecção de bocejo: {e}")
            return False

    def _detectar_micro_cochilo(self, ear):
        """Detecta micro-cochilos e sonolência com maior precisão"""
        # Ajuste de limiares para maior sensibilidade
        EAR_THRESHOLD = 0.21          # Aumentado de 0.18
        MICRO_SLEEP_MIN = 0.3         # Reduzido de 0.5
        ALERTA_THRESHOLD = 0.23       # Novo threshold para alerta precoce

        try:
            if ear < EAR_THRESHOLD:
                if not self.dormindo:
                    self.t_inicial_olhos_fechados = time.time()
                    self.dormindo = True

                tempo_fechado = time.time() - self.t_inicial_olhos_fechados

                # Verificar histórico recente de EAR
                ear_recente = list(self.ear_historico)[-30:]  # último segundo
                ear_medio = np.mean(ear_recente) if ear_recente else ear

                # Detectar padrão de sonolência
                if ear_medio < ALERTA_THRESHOLD and len(ear_recente) > 15:
                    padrao_sonolencia = any(
                        all(e < ALERTA_THRESHOLD for e in ear_recente[i:i+5])
                        for i in range(len(ear_recente)-5)
                    )

                    if padrao_sonolencia:
                        return True, tempo_fechado

                # Micro-cochilo
                if tempo_fechado > MICRO_SLEEP_MIN:
                    return True, tempo_fechado

            else:
                if self.dormindo:
                    tempo_fechado = time.time() - self.t_inicial_olhos_fechados
                    self.dormindo = False

                    if tempo_fechado > MICRO_SLEEP_MIN:
                        self.micro_cochilos += 1

            return False, 0.0

        except Exception as e:
            self.logger.error(f"Erro na detecção de micro-cochilo: {e}")

    def _detectar_inclinacao_cabeca(self, vetores_cabeca):
        """Detecta inclinação da cabeça com maior precisão"""
        vetor_frontal, vetor_lateral, vetor_vertical = vetores_cabeca

        try:
            # Calcular ângulos com maior precisão
            angulo_frontal = np.arccos(
                np.dot(vetor_frontal, [0, 0, 1])) * 180 / np.pi
            angulo_lateral = np.arccos(
                np.dot(vetor_lateral, [1, 0, 0])) * 180 / np.pi
            angulo_vertical = np.arccos(
                np.dot(vetor_vertical, [0, 1, 0])) * 180 / np.pi

            # Usar histórico para reduzir falsos positivos
            self.head_pose_historico.append(
                (angulo_frontal, angulo_lateral, angulo_vertical))

            if len(self.head_pose_historico) >= 10:  # 1/3 segundo em 30 FPS
                # Convert deque to list for array operations
                historico = list(self.head_pose_historico)[-10:]

                # Calcular médias móveis
                ang_front_medio = np.mean([h[0] for h in historico])
                ang_lat_medio = np.mean([h[1] for h in historico])
                ang_vert_medio = np.mean([h[2] for h in historico])

                # Calcular desvios padrão
                ang_front_std = np.std([h[0] for h in historico])
                ang_lat_std = np.std([h[1] for h in historico])
                ang_vert_std = np.std([h[2] for h in historico])

                # Verificar se a inclinação é consistente e significativa
                inclinacao_significativa = (
                    # Aumentado de 30
                    (abs(ang_front_medio - 90) > 35 and ang_front_std < 10) or
                    (abs(ang_lat_medio - 90) > 35 and ang_lat_std < 10) or
                    (abs(ang_vert_medio - 90) > 35 and ang_vert_std < 10)
                )

                if inclinacao_significativa:
                    if not self.cabeca_inclinada:
                        self.t_inicial_cabeca = time.time()
                        self.cabeca_inclinada = True

                    # Aumentar tempo de confirmação
                    if time.time() - self.t_inicial_cabeca > 2.5:  # Reduzido de 3.0
                        return True, (ang_front_medio-90, ang_lat_medio-90, ang_vert_medio-90)
                else:
                    self.cabeca_inclinada = False

            return False, (0, 0, 0)

        except Exception as e:
            self.logger.error(f"Erro na detecção de inclinação: {e}")
            return False, (0, 0, 0)

    # Adicionar função de debug para visualizar ângulos
    def _debug_angulos(self, frame, angulos):
        """Adiciona informações de debug sobre ângulos na tela"""
        angulo_frontal, angulo_lateral, angulo_vertical = angulos
        debug_info = [
            f"Ang. Frontal: {angulo_frontal:.1f}",
            f"Ang. Lateral: {angulo_lateral:.1f}",
            f"Ang. Vertical: {angulo_vertical:.1f}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = frame.shape[0] - 100  # Posicionar na parte inferior

        for i, info in enumerate(debug_info):
            cv2.putText(frame, info, (10, y_pos + i * 30),
                        font, 0.6, (0, 255, 0), 2)

    def _debug_mar(self, frame, mar):
        """Adiciona visualização do MAR atual"""
        altura, largura = frame.shape[:2]
        # Criar gráfico simplificado
        bar_height = 100
        bar_width = 30
        x = largura - 50
        y = altura - 150

        # Normalizar MAR para visualização (0-1)
        mar_norm = min(1.0, mar / 1.0)
        bar_value = int(bar_height * mar_norm)

        # Desenhar barra
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y + bar_height - bar_value),
                      (x + bar_width, y + bar_height),
                      (0, 255, 0), -1)

        # Adicionar texto
        cv2.putText(frame, f"MAR: {mar:.2f}", (x - 70, y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _avaliar_nivel_fadiga(self, ear, mar, piscadas_pm, micro_cochilos, bocejos_10min, inclinacao_cabeca):
        """Avalia o nível de fadiga com maior precisão"""
        try:
            # Pesos ajustados para maior precisão
            pesos = {
                'ear': 0.35,           # Aumentado de 0.3
                'mar': 0.15,           # Aumentado de 0.1
                'piscadas': 0.15,      # Mantido
                'micro_cochilos': 0.20,  # Reduzido de 0.25
                'bocejos': 0.10,       # Mantido
                'inclinacao': 0.05     # Reduzido de 0.1
            }

            # Normalização mais precisa dos valores
            ear_norm = max(0, min(1, 1 - (ear / 0.28)))  # Reduzido de 0.3
            mar_norm = max(0, min(1, mar / 0.65))        # Aumentado de 0.6

            # Normalização das contagens com base em estudos clínicos
            piscadas_norm = max(0, min(1, piscadas_pm / 45))  # Reduzido de 50
            micro_norm = max(0, min(1, micro_cochilos / 3))   # Reduzido de 5
            bocejos_norm = max(0, min(1, bocejos_10min / 2))  # Reduzido de 3

            # Análise do histórico recente
            ear_recente = list(self.ear_historico)[-90:]  # últimos 3 segundos
            if len(ear_recente) > 0:
                ear_tendencia = np.polyfit(
                    range(len(ear_recente)), ear_recente, 1)[0]
                ear_norm = max(ear_norm, min(1, abs(ear_tendencia) * 100))

            # Calcular nível de fadiga
            nivel_fadiga = 100 * (
                pesos['ear'] * ear_norm +
                pesos['mar'] * mar_norm +
                pesos['piscadas'] * piscadas_norm +
                pesos['micro_cochilos'] * micro_norm +
                pesos['bocejos'] * bocejos_norm +
                pesos['inclinacao'] * (1.0 if inclinacao_cabeca else 0.0)
            )

            # Ajuste final baseado em padrões
            if micro_cochilos > 0:
                nivel_fadiga = max(nivel_fadiga, 70)  # Garantir alerta alto

            if ear < 0.21 and np.mean(ear_recente) < 0.23:
                nivel_fadiga = max(nivel_fadiga, 85)  # Alerta crítico

            return nivel_fadiga

        except Exception as e:
            self.logger.error(f"Erro na avaliação de fadiga: {e}")
            return 50.0  # valor médio em caso de erro

    def _gerar_alerta(self, nivel_fadiga, tipo_alerta, dados_adicionais=None):
        """Gera um alerta baseado no nível de fadiga e tipo detectado"""
        tempo_atual = time.time()

        # Verificar intervalo mínimo entre alertas
        if tempo_atual - self.ultimo_alerta < self.config["intervalo_min_alertas"]:
            return None, None  # Return explicit None values instead of just None

        self.ultimo_alerta = tempo_atual
        self.contador_alertas += 1

        # Registrar alerta
        alerta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "nivel_fadiga": nivel_fadiga,
            "tipo": tipo_alerta,
            "dados": dados_adicionais or {}
        }
        self.dados_sessao["alertas"].append(alerta)

        # Log
        self.logger.warning(
            f"ALERTA: {tipo_alerta} - Nível de fadiga: {nivel_fadiga:.1f}")

        # Ajustar intensidade do alerta baseado no nível de fadiga
        if nivel_fadiga > 75:  # Alerta crítico
            cor_alerta = (0, 0, 255)  # Vermelho
            volume = 1.0
            duracao = 1.5
        elif nivel_fadiga > 50:  # Alerta moderado
            cor_alerta = (0, 165, 255)  # Laranja
            volume = 0.7
            duracao = 1.0
        else:  # Alerta leve
            cor_alerta = (0, 255, 255)  # Amarelo
            volume = 0.5
            duracao = 0.5

        # Executar alertas conforme configurado
        if self.config["alerta_sonoro"]:
            try:
                self.som_alerta.set_volume(volume)
                self.som_alerta.play()
            except Exception as e:
                self.logger.error(f"Erro ao tocar alerta sonoro: {e}")

        return cor_alerta, duracao

    def _analisar_anomalias(self, ear, mar, orientacao_cabeca):
        """Detecta padrões anômalos usando machine learning"""
        # Extrair características
        inclinacao_frontal, inclinacao_lateral, inclinacao_vertical = orientacao_cabeca

        # Criar vetor de características
        X = np.array([[
            ear,
            mar,
            abs(inclinacao_frontal),
            abs(inclinacao_lateral),
            abs(inclinacao_vertical),
            # piscadas por segundo nos últimos 60s
            len(self.historico_piscadas) / 60,
            self.micro_cochilos
        ]])

        # Adicionar aos dados de treinamento se configurado
        if self.config["coletar_dados_treinamento"]:
            self.dados_treinamento.append(X[0])

        # Detectar anomalias apenas se o modelo já foi treinado
        if hasattr(self.anomaly_detector, 'offset_'):
            # -1 para anomalias, 1 para normal
            resultado = self.anomaly_detector.predict(X)
            score = self.anomaly_detector.score_samples(X)[0]

            if resultado[0] == -1:
                return True, score

        return False, 0.0

    def _treinar_modelo_anomalias(self):
        """Treina o modelo de detecção de anomalias com dados coletados"""
        if len(self.dados_treinamento) > 100:  # Treinar apenas com dados suficientes
            X_train = np.array(self.dados_treinamento)
            self.anomaly_detector.fit(X_train)

            # Salvar modelo
            modelo_path = os.path.join(
                self.config["pasta_modelos"], "anomaly_detector.joblib")
            joblib.dump(self.anomaly_detector, modelo_path)
            self.logger.info(
                f"Modelo de detecção de anomalias treinado e salvo em {modelo_path}")

            # Limpar buffer de treinamento
            self.dados_treinamento = []

    def _analisar_tendencias_thread(self):
        """Thread para análise de tendências de longo prazo"""
        while not self.stop_threads:
            # Executar apenas se há dados suficientes
            if len(self.ear_historico) > 60:
                try:
                    # Calcular tendências
                    ear_media = np.mean(self.ear_historico)
                    ear_tendencia = np.polyfit(
                        range(len(self.ear_historico)), self.ear_historico, 1)[0]

                    # Detectar fadiga progressiva (EAR diminuindo constantemente)
                    if ear_tendencia < -0.0005 and ear_media < 0.25:
                        self.logger.warning(
                            "Tendência de fadiga progressiva detectada!")
                        # Aqui poderíamos gerar um alerta de longo prazo

                except Exception as e:
                    self.logger.error(f"Erro na análise de tendências: {e}")

            # Dormir por 30 segundos
            time.sleep(30)

    def _salvar_dados_periodicamente(self):
        """Thread para salvar dados de sessão periodicamente"""
        while not self.stop_threads:
            try:
                # Atualizar métricas acumuladas
                self.dados_sessao["metricas"]["piscadas_total"] = self.contagem_piscadas
                self.dados_sessao["metricas"]["bocejos_total"] = self.contagem_bocejos
                self.dados_sessao["metricas"]["micro_cochilos"] = self.micro_cochilos
                self.dados_sessao["metricas"]["duracao_total"] = time.time(
                ) - self.inicio_sessao
                self.dados_sessao["metricas"]["alertas_total"] = self.contador_alertas

                # Salvar dados da sessão
                nome_arquivo = os.path.join(
                    self.config["pasta_dados"],
                    f"sessao_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

                with open(nome_arquivo, 'w') as f:
                    json.dump(self.dados_sessao, f, indent=4)

                # Treinar modelo se houver dados suficientes
                if self.config["coletar_dados_treinamento"]:
                    self._treinar_modelo_anomalias()

                # Dormir por 5 minutos
                time.sleep(300)

            except Exception as e:
                self.logger.error(f"Erro ao salvar dados: {e}")
                time.sleep(60)  # Tentar novamente em 1 minuto

    def executar(self):
        """Executa o detector de fadiga"""
        pygame.init()
        pygame.mixer.init()

        self.cap = cv2.VideoCapture(self.config["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["resolucao"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["resolucao"][1])

        if not self.cap.isOpened():
            self.logger.error("Erro ao abrir câmera")
            return

        self.logger.info("Iniciando detector de fadiga avançado...")

        # Variáveis para cálculo de FPS e métricas
        tempo_anterior = time.time()
        contador_frames = 0
        fps = 0

        # Variáveis para cálculo de piscadas por minuto
        tempo_inicio_piscadas = time.time()
        piscadas_ultimo_minuto = 0
        ear_anterior = 0.3  # valor inicial aproximado
        piscada_em_andamento = False

        with self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            while self.cap.isOpened():
                sucesso, frame = self.cap.read()
                if not sucesso:
                    self.logger.warning("Frame não capturado")
                    continue

                # Calcular FPS
                contador_frames += 1
                if (time.time() - tempo_anterior) > 1.0:
                    fps = contador_frames
                    contador_frames = 0
                    tempo_anterior = time.time()

                # Processar frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = face_mesh.process(frame_rgb)

                # Criar overlay para informações
                overlay = frame.copy()

                if resultados.multi_face_landmarks:
                    for face_landmarks in resultados.multi_face_landmarks:
                        # Extrair métricas
                        ear = self._calculo_ear(
                            face_landmarks.landmark, self.p_olho_dir, self.p_olho_esq)
                        mar = self._calculo_mar(
                            face_landmarks.landmark, self.p_boca)
                        vetores_cabeca = self._calculo_orientacao_cabeca(
                            face_landmarks.landmark, self.p_cabeca)

                        # Detectar piscada
                        if ear_anterior > 0.25 and ear < 0.25 and not piscada_em_andamento:
                            piscadas_ultimo_minuto += 1
                            piscada_em_andamento = True
                        elif ear > 0.25:
                            piscada_em_andamento = False

                        ear_anterior = ear

                        # Detectar eventos
                        bocejo = self._detectar_bocejo(mar, ear)
                        micro_cochilo, tempo_fechado = self._detectar_micro_cochilo(
                            ear)
                        inclinacao, angulos = self._detectar_inclinacao_cabeca(
                            vetores_cabeca)

                        # Análise de anomalias
                        anomalia, score = self._analisar_anomalias(
                            ear, mar, angulos)

                        # Calcular nível de fadiga
                        nivel_fadiga = self._avaliar_nivel_fadiga(
                            ear, mar,
                            piscadas_ultimo_minuto,
                            self.micro_cochilos,
                            self.contagem_bocejos,
                            inclinacao
                        )

                        # Desenhar métricas na tela
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        y_pos = 30
                        linha_altura = 30

                        # Informações principais
                        metricas = [
                            f"FPS: {fps}",
                            f"EAR: {ear:.2f}",
                            f"MAR: {mar:.2f}",
                            f"Nivel Fadiga: {nivel_fadiga:.1f}%",
                            f"Piscadas/min: {piscadas_ultimo_minuto}",
                            f"Micro-cochilos: {self.micro_cochilos}",
                            f"Bocejos: {self.contagem_bocejos}",
                        ]

                        # Adicionar alertas específicos
                        if micro_cochilo:
                            metricas.append(
                                f"ALERTA: Olhos fechados por {tempo_fechado:.1f}s")
                        if bocejo:
                            metricas.append("ALERTA: Bocejo detectado")
                        if inclinacao:
                            metricas.append("ALERTA: Cabeça inclinada")

                        # Desenhar fundo semi-transparente para métricas
                        cv2.rectangle(overlay, (10, 10), (300, y_pos + linha_altura * len(metricas)),
                                      (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                        # Desenhar métricas
                        for i, metrica in enumerate(metricas):
                            cv2.putText(frame, metrica, (20, y_pos + i * linha_altura),
                                        font, 0.6, (255, 255, 255), 2)

                        # Gerar alertas se necessário
                        if nivel_fadiga > 50 or anomalia:
                            cor_alerta, duracao = self._gerar_alerta(
                                nivel_fadiga,
                                "Fadiga detectada" if nivel_fadiga > 50 else "Comportamento anormal",
                                {
                                    "ear": ear,
                                    "mar": mar,
                                    "angulos": angulos,
                                    "anomalia_score": score if anomalia else None
                                }
                            )

                            if cor_alerta is not None and duracao is not None:  # Check if alert was actually generated
                                cv2.rectangle(frame, (0, 0),
                                              (frame.shape[1], frame.shape[0]),
                                              cor_alerta, 3)

                        # Atualizar históricos
                        self.ear_historico.append(ear)
                        self.mar_historico.append(mar)

                        # Desenhar landmarks faciais
                        self.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                        )

                # Resetar contagem de piscadas a cada minuto
                if time.time() - tempo_inicio_piscadas > 60:
                    tempo_inicio_piscadas = time.time()
                    piscadas_ultimo_minuto = 0

                # Mostrar frame
                cv2.imshow("Detector de Fadiga Avançado", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Limpeza
        self.stop_threads = True
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        self.logger.info("Detector de fadiga encerrado")

    def __del__(self):
        """Destrutor da classe"""
        self.stop_threads = True
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    # Exemplo de uso
    detector = DetectorFadigaAvancado()
    detector.config["modo_debug"] = True
    try:
        detector.executar()
    except KeyboardInterrupt:
        print("\nDetector encerrado pelo usuário")
    finally:
        detector.__del__()
