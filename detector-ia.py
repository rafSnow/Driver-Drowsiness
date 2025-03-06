import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
import logging
import time
import os
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='detector_sonolencia_log.txt'
)

# Carregar o modelo
logging.info("Carregando modelo...")
try:
    modelo = tf.keras.models.load_model('my_model.h5')
    logging.info(f"Modelo carregado com sucesso. Formato de entrada: {modelo.input_shape}")
except Exception as e:
    logging.error(f"Erro ao carregar modelo: {e}")
    print(f"Erro ao carregar modelo: {e}")
    exit(1)

# Configurações de depuração
modo_debug = True
benchmark_mode = False
benchmark_frames = 100
benchmark_results = []
previsoes = []
frame_counter = 0

# Criar diretório para frames de depuração
os.makedirs("frames_debug", exist_ok=True)

# Configurar MediaPipe para detecção facial
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Função para testar imagens de controle
def testar_imagens_controle():
    # Imagens de teste (substitua pelos caminhos reais)
    caminhos_teste = [
        "test_images/alerta1.jpg",
        "test_images/alerta2.jpg",
        "test_images/sonolento1.jpg",
        "test_images/sonolento2.jpg"
    ]

    print("Testando com imagens de controle:")
    for caminho in caminhos_teste:
        try:
            if not os.path.exists(caminho):
                print(f"Arquivo não existe: {caminho}")
                continue

            img = cv2.imread(caminho)
            if img is None:
                print(f"Não foi possível carregar: {caminho}")
                continue

            # Processar como faria com frames da webcam
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(img_rgb)

            if resultados.multi_face_landmarks:
                for face_landmarks in resultados.multi_face_landmarks:
                    # Obter coordenadas da face
                    h, w, c = img.shape
                    x_min, y_min, x_max, y_max = w, h, 0, 0

                    # Encontrar os limites da face
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    # Adicionar margem
                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    # Recortar a face
                    face = img[y_min:y_max, x_min:x_max]
                    if face.size == 0:
                        print(f"Face recortada inválida para: {caminho}")
                        continue

                    # Redimensionar e preparar para o modelo
                    face_redimensionada = cv2.resize(face, (145, 145))
                    face_normalizada = face_redimensionada / 255.0
                    face_batch = np.expand_dims(face_normalizada, axis=0)

                    # Fazer a previsão
                    resultado = modelo.predict(face_batch, verbose=0)

                    # Interpretar resultado
                    estado = "Sonolento" if resultado[0][0] > 0.5 else "Alerta"

                    # Desenhar retângulo e texto
                    cor = (0, 0, 255) if estado == "Sonolento" else (0, 255, 0)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), cor, 2)
                    cv2.putText(img, f"{estado}: {resultado[0][0]:.2f}", (x_min, y_min-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
            else:
                print(f"Nenhuma face detectada em: {caminho}")

            # Mostrar resultados por mais tempo
            cv2.imshow(f'Teste Controle - {os.path.basename(caminho)}', img)
            cv2.waitKey(0)  # Esperar tecla

        except Exception as e:
            print(f"Erro ao processar {caminho}: {e}")
            import traceback
            print(traceback.format_exc())

    cv2.destroyAllWindows()
    print("Teste de controle finalizado")

# Descomente a linha abaixo para executar testes com imagens estáticas
# testar_imagens_controle()

# Iniciar captura de vídeo
logging.info("Iniciando captura de vídeo...")
cap = cv2.VideoCapture(0)  # 0 para webcam padrão

if not cap.isOpened():
    logging.error("Não foi possível abrir a webcam")
    print("Erro: Não foi possível abrir a webcam")
    exit(1)

# Função para criar e atualizar gráfico de histórico
def atualizar_grafico_historico():
    if len(previsoes) < 2:
        return None

    hist_width = 300
    hist_height = 150
    hist_img = np.ones((hist_height, hist_width, 3), dtype=np.uint8) * 255

    # Desenhar linha do limiar (0.5)
    cv2.line(hist_img, (0, int(hist_height/2)),
            (hist_width, int(hist_height/2)), (0, 0, 0), 1)

    # Desenhar valores de previsão
    max_pontos = min(hist_width, len(previsoes))
    for i in range(1, max_pontos):
        # Valores normalizados para o tamanho do gráfico
        y1 = int(hist_height - (previsoes[-(i)] * hist_height))
        y2 = int(hist_height - (previsoes[-(i+1)] * hist_height))

        # Posição x (da direita para a esquerda)
        x1 = hist_width - i
        x2 = hist_width - (i+1)

        # Cor baseada no valor (sonolento = vermelho, alerta = verde)
        cor = (0, 0, 255) if previsoes[-(i)] > 0.5 else (0, 255, 0)

        # Desenhar linha
        cv2.line(hist_img, (x1, y1), (x2, y2), cor, 2)

    # Desenhar linhas de grade
    for i in range(0, 11, 2):
        y = int(hist_height * (1 - i/10))
        cv2.line(hist_img, (0, y), (hist_width, y), (200, 200, 200), 1)
        cv2.putText(hist_img, f"{i/10:.1f}", (5, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return hist_img

# Loop principal
logging.info("Iniciando loop principal de captura e processamento...")
try:
    while cap.isOpened():
        # Iniciar timer para cálculo de FPS
        start_time = time.time()
        frame_counter += 1

        try:
            # Capturar frame
            success, imagem = cap.read()
            if not success:
                logging.error("Falha ao ler frame da câmera")
                break

            # Guardar cópia original
            imagem_original = imagem.copy()

            # Converter para RGB (MediaPipe usa RGB)
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

            # Iniciar timer para benchmark do processamento
            if benchmark_mode and len(benchmark_results) < benchmark_frames:
                start_process = time.time()

            # Detectar faces
            resultados = face_mesh.process(imagem_rgb)

            face_detectada = False
            if resultados.multi_face_landmarks:
                face_detectada = True
                logging.info(f"Frame {frame_counter}: Face detectada")

                for face_landmarks in resultados.multi_face_landmarks:
                    # Obter coordenadas da face
                    h, w, c = imagem.shape
                    x_min, y_min, x_max, y_max = w, h, 0, 0

                    # Encontrar os limites da face
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    # Adicionar margem
                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    # Se quiser visualizar os pontos do rosto
                    if modo_debug:
                        # Desenhar a malha facial
                        mp_drawing.draw_landmarks(
                            image=imagem,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )

                    # Recortar a face
                    face = imagem_original[y_min:y_max, x_min:x_max]
                    if face.size == 0:
                        logging.warning(f"Frame {frame_counter}: Face recortada com tamanho zero")
                        continue

                    # Mostrar a face recortada em modo debug
                    if modo_debug:
                        cv2.imshow('Face Recortada', face)

                    # Redimensionar e preparar para o modelo
                    face_redimensionada = cv2.resize(face, (145, 145))
                    face_normalizada = face_redimensionada / 255.0
                    face_batch = np.expand_dims(face_normalizada, axis=0)

                    # Mostrar a face redimensionada em modo debug
                    if modo_debug:
                        cv2.imshow('Face Redimensionada', face_redimensionada)

                    # Fazer a previsão
                    resultado = modelo.predict(face_batch, verbose=0)
                    previsao = resultado[0][0]
                    previsoes.append(previsao)

                    # Manter apenas as últimas 100 previsões
                    if len(previsoes) > 100:
                        previsoes.pop(0)

                    # Estatísticas a cada 30 frames
                    if len(previsoes) % 30 == 0 and len(previsoes) > 0:
                        media = sum(previsoes) / len(previsoes)
                        logging.info(f"Média de previsões: {media:.3f}, Min: {min(previsoes):.3f}, Max: {max(previsoes):.3f}")

                    # Interpretar resultado
                    estado = "Sonolento" if previsao < 0.4 else "Alerta"

                    # Desenhar retângulo e texto
                    cor = (0, 0, 255) if estado == "Sonolento" else (0, 255, 0)
                    cv2.rectangle(imagem, (x_min, y_min), (x_max, y_max), cor, 2)
                    cv2.putText(imagem, f"{estado}: {previsao:.2f}", (x_min, y_min-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

                    # Salvar frames onde há mudança de estado
                    if len(previsoes) > 1 and ((previsoes[-2] < 0.5 and previsoes[-1] >= 0.5) or
                                              (previsoes[-2] >= 0.5 and previsoes[-1] < 0.5)):
                        timestamp = int(time.time())
                        cv2.imwrite(f"frames_debug/mudanca_{timestamp}_{previsoes[-1]:.3f}.jpg", imagem)
                        cv2.imwrite(f"frames_debug/face_{timestamp}_{previsoes[-1]:.3f}.jpg", face)
                        logging.info(f"Mudança de estado detectada: {previsoes[-2]:.3f} -> {previsoes[-1]:.3f}")
            else:
                logging.info(f"Frame {frame_counter}: Nenhuma face detectada")

            # Finalizar timer para benchmark
            if benchmark_mode and len(benchmark_results) < benchmark_frames:
                process_time = time.time() - start_process
                benchmark_results.append(process_time)

                if len(benchmark_results) == benchmark_frames:
                    avg_time = sum(benchmark_results) / len(benchmark_results)
                    logging.info(f"Benchmark concluído: Média de {avg_time*1000:.2f}ms por frame")
                    logging.info(f"FPS médio em processamento: {1.0/avg_time:.2f}")

            # Calcular FPS
            fps = 1.0 / (time.time() - start_time)

            # Exibir informações de depuração na tela
            if modo_debug:
                info_text = []
                info_text.append(f"FPS: {fps:.1f}")
                info_text.append(f"Frame: {frame_counter}")

                if face_detectada:
                    info_text.append(f"Face: Sim")
                    if 'previsao' in locals():
                        info_text.append(f"Previsão: {previsao:.3f}")
                        info_text.append(f"Estado: {estado}")
                else:
                    info_text.append(f"Face: Não")

                # Exibir texto na parte inferior da tela
                for i, text in enumerate(info_text):
                    cv2.putText(imagem, text, (10, imagem.shape[0] - 10 - i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Desenhar gráfico histórico
            if modo_debug and len(previsoes) > 1:
                hist_img = atualizar_grafico_historico()
                if hist_img is not None:
                    cv2.imshow('Histórico de Previsões', hist_img)

            # Mostrar imagem
            cv2.imshow('Detecção de Sonolência', imagem)

            # Pressionar 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Encerrando por comando do usuário (tecla 'q')")
                break

        except Exception as e:
            logging.error(f"Erro no processamento do frame {frame_counter}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Continuar para o próximo frame
            continue

except KeyboardInterrupt:
    logging.info("Encerrando por interrupção do teclado")
except Exception as e:
    logging.error(f"Erro inesperado: {e}")
    import traceback
    logging.error(traceback.format_exc())
finally:
    # Liberar recursos
    logging.info("Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Programa encerrado")
    print("Programa encerrado")