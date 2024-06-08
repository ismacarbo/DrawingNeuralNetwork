import pygame
import numpy as np
from keras.models import load_model

pygame.init()

dimensioneFinestra = 500
win = pygame.display.set_mode((dimensioneFinestra, dimensioneFinestra))
pygame.display.setCaption("Disegna un oggetto")

black = (0, 0, 0)
gray = (200, 200, 200)
white = (255, 255, 255)

# Load the model
model = load_model('object_model.keras')

categorie = ['airplane', 'apple', 'axe', 'cat', 'car'] #categorie oggetti (espandibile)

font = pygame.font.SysFont(None, 24)

def predict_object(image):
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict([image])[0]
    return categorie[np.argmax(prediction)], max(prediction)

def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

def main():
    run = True
    disegno = False
    raggio = 2  
    nomeOggetto = ""
    accuratezza = 0

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                disegno = True
            if event.type == pygame.MOUSEBUTTONUP:
                disegno = False
                img_array = pygame.surfarray.array3d(win)
                img_array = img_array.swapaxes(0, 1)
                gray_image = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
                gray_image = pygame.transform.scale(pygame.surfarray.make_surface(gray_image), (28, 28))
                gray_image = pygame.surfarray.array3d(gray_image).swapaxes(0, 1)
                gray_image = np.dot(gray_image[..., :3], [0.2989, 0.5870, 0.1140])
                
                nomeOggetto, accuratezza = predict_object(gray_image)
                accuratezza = round(accuratezza, 2)

            if event.type == pygame.MOUSEMOTION:
                if disegno:
                    mouseX, mouseY = event.pos
                    pygame.draw.circle(win, white, (mouseX, mouseY), raggio)
        
        
        pygame.draw.rect(win, gray, (0, 0, dimensioneFinestra, 60))  
        draw_text(f"Oggetto Riconosciuto: {nomeOggetto}", font, black, win, 10, 10)
        draw_text(f"Accuratezza: {accuratezza}", font, black, win, 10, 30)

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    win.fill(black)
    main()
