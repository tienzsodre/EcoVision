"""
YOLO + Webcam (sem Tkinter)

Fluxo: Webcam -> Frame -> YOLO -> Detecção -> Exibição
Pressione a tecla 'q' na janela de vídeo para encerrar.
"""

import cv2
from ultralytics import YOLO

MODELO = "yolo11n.pt"
INDICE_CAMERA = 1  # trocar pra 1 se abrir a camera errada


def main():

    model = YOLO(MODELO)

    cap = cv2.VideoCapture(INDICE_CAMERA)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    print("Webcam aberta. Pressione 'q' para sair.")

    try:
        while True:

            ok, frame = cap.read()
            if not ok:
                print("Erro: falha ao ler o frame da câmera.")
                break

            resultados = model.predict(frame, verbose=False)

            frame_anotado = resultados[0].plot()

            cv2.imshow("YOLO + Webcam", frame_anotado)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Câmera liberada. Programa encerrado.")


if __name__ == "__main__":
    main()
