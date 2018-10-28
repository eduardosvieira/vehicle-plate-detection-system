import cv2

filename = "placa.png"

img = cv2.imread(filename, 0)

blur = cv2.GaussianBlur(img,(5,5),0)

edges = cv2.Canny(blur, 100, 200)

cv2.imshow("Imagem em Escala de Cinza", img)
cv2.imshow("Imagem em filtro Gausssino", blur)
cv2.imshow("Usando Operador Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
