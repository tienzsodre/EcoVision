import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

LARGURA, ALTURA = 640, 480


class App:
    def __init__(self, janela):
        self.janela = janela
        janela.title("Detector de Objetos")
        janela.geometry("800x650")
        janela.protocol("WM_DELETE_WINDOW", self.fechar)
        janela.resizable(False, False)


        tk.Label(janela, text="Detector de Objetos", font=("Arial", 18)).pack(pady=10)

        # Area da camera
        self.canvas = tk.Canvas(janela, width=LARGURA, height=ALTURA, bg="gray20", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.retangulo = self.canvas.create_rectangle(2, 2, LARGURA - 2, ALTURA - 2, outline="gray50", width=3)
        self.imagem_no_canvas = None

        # Botoes
        frame_botoes = tk.Frame(janela)
        frame_botoes.pack(pady=10)

        tk.Button(frame_botoes, text="Iniciar", width=12, command=self.iniciar).pack(side="left", padx=5)
        tk.Button(frame_botoes, text="Parar", width=12, command=self.parar).pack(side="left", padx=5)
        tk.Button(frame_botoes, text="Sair", width=12, command=self.fechar).pack(side="left", padx=5)

        # Barra de status
        self.status = tk.Label(janela, text="Status: parado", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

        self.cap = None
        self.rodando = False
        self.model = None

    def iniciar(self):
        if self.rodando:
            return

        self.status.config(text="Status: carregando modelo...")
        self.janela.update()  # força atualizar a tela antes de travar carregando o modelo
        if self.model is None:
            self.model = YOLO("yolov8n.pt")

        self.cap = cv2.VideoCapture(1)  # trocar pra 1 se abrir a camera errada
        if not self.cap.isOpened():
            self.status.config(text="Status: não consegui abrir a câmera")
            self.cap = None
            return

        self.canvas.itemconfig(self.retangulo, outline="lime")
        self.status.config(text="Status: rodando")
        self.rodando = True
        self.atualizar_frame()

    def parar(self):
        self.rodando = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.canvas.itemconfig(self.retangulo, outline="gray50")
        self.status.config(text="Status: parado")

    def atualizar_frame(self):
        if not self.rodando or self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok:
            # comentar essas 2 linhas so pra camera
            resultados = self.model(frame, verbose=False)
            frame = resultados[0].plot()

            frame = cv2.resize(frame, (LARGURA, ALTURA))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            if self.imagem_no_canvas is None:
                self.imagem_no_canvas = self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
            else:
                self.canvas.itemconfig(self.imagem_no_canvas, image=img_tk)
            self.canvas.imgtk = img_tk  # evita o garbage collector apagar

            self.canvas.tag_raise(self.retangulo)


        self.janela.after(15, self.atualizar_frame)

    def fechar(self):
        self.parar()
        self.janela.destroy()


if __name__ == "__main__":
    janela = tk.Tk()
    App(janela)
    janela.mainloop()