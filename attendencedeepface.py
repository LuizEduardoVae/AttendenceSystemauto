import cv2 as cv
from retinaface import RetinaFace

# Função para processar cada frame e detectar rostos usando RetinaFace
def process_frame(image_read, scale=0.5):
    """
    Processa o frame para detectar rostos usando o RetinaFace.

    Args:
        image_read (ndarray): O frame atual capturado da webcam.
        scale (float): Fator de escala para redimensionar o frame para processamento mais rápido. Default é 0.5.

    Returns:
        image_read (ndarray): O frame processado com retângulos desenhados ao redor dos rostos detectados.
    """
    # Redimensiona o frame para acelerar o processamento
    image_resized = cv.resize(image_read, (0, 0), fx=scale, fy=scale)
    image_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)

    # Detecta rostos usando RetinaFace
    faces = RetinaFace.detect_faces(image_rgb)

    if faces:
        for key, face in faces.items():
            # Obtém as coordenadas da área facial detectada
            x1, y1, x2, y2 = face['facial_area']

            # Ajusta as coordenadas de volta para a escala original
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

            # Desenha um retângulo ao redor do rosto detectado
            cv.rectangle(image_read, (x1, y1), (x2, y2), (50, 200, 50), 2)
            cv.putText(image_read, "Face", (x1, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 2)
    else:
        print("No face detected")

    return image_read


# Função principal para iniciar a captura de vídeo e o processo de detecção de rostos
def main():
    """
    A função principal inicializa a captura de vídeo da webcam e processa cada frame para detectar rostos.
    """
    # Inicia a captura de vídeo
    capture = cv.VideoCapture(1)  # Use o índice 0 para a webcam

    while True:
        result, image_read = capture.read()
        if result:
            flipped = cv.flip(image_read, 1)  # Inverte o frame horizontalmente para melhor visualização
            processed_frame = process_frame(flipped)  # Processa o frame para detectar rostos
            cv.imshow("Webcam - Face Detection", processed_frame)  # Exibe o frame processado na janela

        if cv.waitKey(1) & 0xFF == 27:  # Pressione 'ESC' para sair
            break

    # Libera a captura de vídeo e fecha todas as janelas
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()