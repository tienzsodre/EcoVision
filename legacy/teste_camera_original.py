import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

print("Câmera aberta!")
print("Pressione Q para sair.")

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Não foi possível capturar o frame.")
        break

    cv2.imshow("Teste da Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()