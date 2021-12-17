import pygame
import math
from pygame.font import Font
import numpy as np
import sys
import tensorflow as tf

pygame.init()

# set features for pygame
size = width, height = 500, 500
gray = (224, 224, 224)
black = (0, 0, 0)
white = (255, 255, 255)
PIXEL_SIZE = math.floor(4 / 5 * width / 28)
text_font = pygame.font.Font('Ubuntu-Bold.ttf', 30)
screen = pygame.display.set_mode(size)


def new_grid():
    img_array = []
    grid = []
    for i in range(28):
        row = []
        row_array = []
        for j in range(28):
            row.append(white)
            row_array.append([0.])
        grid.append(row)
        img_array.append(row_array)

    return grid, img_array


def draw_grid(grid):
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            pygame.draw.rect(screen, value, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))


def drawing_pad():
    grid, img_array = new_grid()
    dragging = False
    done = False
    num_answer = ""

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            screen.fill(black)

            draw_grid(grid)

            # placement for number prediction
            num = text_font.render(num_answer, True, white)
            predict_rect = num.get_rect()
            predict_rect.center = ((width * 8 / 10), (height / 8))
            screen.blit(num, predict_rect)

            # buttons
            button_width = width/5
            button_height = height/15

            submit_button = pygame.Rect((width / 8), (height * 5/ 6), button_width, button_height)
            submit_text = text_font.render("Predict", True, white)
            submit_rect = submit_text.get_rect()
            submit_rect.center = submit_button.center
            pygame.draw.rect(screen, gray, submit_button)
            screen.blit(submit_text, submit_rect)

            clear_button = pygame.Rect((width / 8), (height * 5 / 6) + button_height + 5, button_width, button_height)
            clear_text = text_font.render("Clear", True, white)
            clear_rect = clear_text.get_rect()
            clear_rect.center = clear_button.center
            pygame.draw.rect(screen, gray, clear_button)
            screen.blit(clear_text, clear_rect)

            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                if submit_button.collidepoint(mouse_pos):
                    done = True
                elif clear_button.collidepoint(mouse_pos):
                    grid, img_array = new_grid()

            # events for drawing
            if event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True

            if event.type == pygame.MOUSEBUTTONUP:
                dragging = False

            if event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_pos = pygame.mouse.get_pos()
                    start_j = math.floor(mouse_pos[0] / PIXEL_SIZE)
                    start_i = math.floor(mouse_pos[1] / PIXEL_SIZE)

                    step = 1
                    for i in range(start_i - step, start_i + step):
                        for j in range(start_j - step, start_j + step):
                            if 0 <= i < 28 and 0 <= j < 28:
                                grid[i][j] = black
                                img_array[i][j] = [1.]

            pygame.display.update()

        if done:
            model = tf.keras.models.load_model(sys.argv[1])
            prediction = predict(model, np.asarray([img_array]))
            num_answer = str(prediction)
            done = False


def predict(model, img_array):
    predictions = model.predict(img_array)[0]
    print(predictions)

    largest = 0
    index = 0
    for i in range(10):
        if predictions[i] * 100 > largest:
            largest = predictions[i] * 100
            index = i

    return index


def main():
    if len(sys.argv) != 2:
        print('Use like this: python write.py [model file]')
        return

    drawing_pad()


if __name__ == '__main__':
    main()
