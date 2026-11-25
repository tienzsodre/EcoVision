import tkinter as tk
 
 
class App:
    def __init__(self, janela):
        self.janela = janela
        janela.title("Detector de Objetos")
        janela.geometry("800x600")
        janela.resizable(False, False)
 
        # titulo
        tk.Label(janela, text="Detector de Objetos", font=("Arial", 18)).pack(pady=10)
 
        # area da camera
        self.canvas = tk.Canvas(janela, width=640, height=480, bg="gray20", highlightthickness=0)
        self.canvas.pack(pady=10)


        self.canvas.create_rectangle(10, 10, 630, 470, outline="gray20" , width=3) 
 
        # botões
        frame_botoes = tk.Frame(janela)
        frame_botoes.pack(pady=10)
 
        tk.Button(frame_botoes, text="Iniciar", width=12, command=self.iniciar).pack(side="left", padx=5)
        tk.Button(frame_botoes, text="Parar", width=12, command=self.parar).pack(side="left", padx=5)
        tk.Button(frame_botoes, text="Sair", width=12, command=janela.destroy).pack(side="left", padx=5)
 
        # Barra de status
        self.status = tk.Label(janela, text="Status: parado", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 10))
 
    def iniciar(self):
        # TODO: abrir a câmera e começar a detecção com YOLO
        self.area_video.config(text="Câmera ligada (placeholder)")
        self.status.config(text="Status: rodando")
 
    def parar(self):
        # TODO: parar a câmera e liberar recursos
        self.area_video.config(text="Câmera desligada")
        self.status.config(text="Status: parado")

    def sair(self):
        self.janela.destroy()
 
 
if __name__ == "__main__":
    janela = tk.Tk()
    App(janela)
    janela.mainloop()
 