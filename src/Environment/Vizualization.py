import pygame
import numpy as np
from src.Environment.Grid import GridMap
from src.Environment.State import Position


class Vizualization:
    def __init__(self):
        self.window = None
        self.clock = None
        self.window_size = 512

    def render(self, mapa):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        a, agent_positions = graph_to_RGB_array(mapa)
        size = a.shape[1]
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 255, 0))
        pix_square_size = (
                self.window_size / size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                pygame.draw.rect(
                    canvas,
                    a[:, i, j],
                    pygame.Rect(
                        pix_square_size * np.array([j, i]),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # Finally, add some gridlines
        for x in range(a.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            for x in range(a.shape[1] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )

        # Now we draw the agent
        for position in agent_positions:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (np.flip(np.array(position)) + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(a.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            for y in range(a.shape[2] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * y, 0),
                    (pix_square_size * y, self.window_size),
                    width=3,
                )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4)

    def render_center(self, a_):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        a, agent_positions = graph_to_RGB_array(a_)
        size = a.shape[1]
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 255, 0))
        pix_square_size = (
                self.window_size / size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                pygame.draw.rect(
                    canvas,
                    a[:, i, j],
                    pygame.Rect(
                        pix_square_size * np.array([j, i]),
                        (pix_square_size, pix_square_size),
                    ),
                )

        for position in agent_positions:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (np.flip(np.array(position)) + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(a.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            for x in range(a.shape[1] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4)


def graph_to_RGB_array(a):
    rgb = np.zeros((3, a.shape[1], a.shape[2]), dtype=np.uint8)
    agents = []
    colors = [[255, 0, 0], [0, 0, 0], [255, 255, 255]]
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            if a[0, i, j] > 0:
                rgb[:, i, j] = [255, 0, 0]
            elif a[0, i, j] == 0 and a[1, i, j] == 0 and a[2, i, j] == 0:
                rgb[:, i, j] = [0, 0, 255]
            elif a[1, i, j] == 255:
                rgb[:, i, j] = [0, 0, 0]
            elif a[2, i, j] == 255:
                rgb[:, i, j] = [255, 255, 255]
            else:
                counter = 0
                for k in range(3):
                    if a[k, i, j] > 0:
                        counter += 1
                    rgb[:, i, j] += np.array((np.array(colors[k]) * int(a[k, i, j])/255),dtype=np.uint8)
                rgb[:,i,j]/counter
            if a[3, i, j] == 255:
                agents.append((i, j))
    return rgb, agents
