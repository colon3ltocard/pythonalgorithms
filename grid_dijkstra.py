"""
Builds a grid with obstacles and animate
bidirectionnal A* versus bidirectionnal Dijkstra
"""
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod

from dijkstra import dijkstra, Node, plot_graph


class Obstacle(ABC):
    @abstractmethod
    def collides(self, x: float, y: float) -> bool:
        """
        Returns True if the Node at (x,y)
        collides with this obstacle.
        """


@dataclass
class Circle(Obstacle):
    """
    A circle of center (x,y) and radius r
    """

    x: float
    y: float
    r: float

    def collides(self, x: float, y: float) -> bool:
        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) <= self.r


@dataclass
class Rectangle(Obstacle):
    x: float
    y: float
    w: float
    h: float

    def collides(self, x: float, y: float) -> bool:
        return (
            self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h
        )


@dataclass
class Ring(Obstacle):
    """
    A Ring centered at (x,y)
    with inner radius ri and outter radius ro
    thetat_start and theta_end are the start and end
    angle in radians of the ring "opening" to the world
    """

    x: float
    y: float
    ri: float
    ro: float
    theta_start: float
    theta_end: float

    def collides(self, x: float, y: float) -> bool:
        d = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        a = math.atan2(y - self.y, x - self.x)
        return (
            self.ri <= d <= self.ro
            and not self.theta_start < a < self.theta_end
        )


@dataclass
class Grid:
    """
    A grid of nodes with obstacles.
    It behaves like a list of Nodes but builds them on the fly
    as requested.
    """

    x_min: float
    x_max: float
    delta_x: float

    y_min: float
    y_max: float
    delta_y: float

    obstacles: list[Obstacle]

    def __post_init__(self) -> None:
        self.nodes_per_line = math.ceil(
            (self.x_max - self.x_min) / self.delta_x
        )
        self.no_lines = math.ceil((self.y_max - self.y_min) / self.delta_y)

        # we compute internode distances once and for all
        self.deltas = dict()
        self.deltas[(1, 0)] = self.delta_x
        self.deltas[(0, 1)] = self.delta_y
        self.deltas[(1, 1)] = math.sqrt(self.delta_x ** 2 + self.delta_y ** 2)

    def __getitem__(self, index):

        if not (-len(self) < index < len(self)):
            raise IndexError(
                f'{index} is out of range ]{-len(self)},{len(self)}['
            )

        if index < 0:
            index = len(self) + index

        y_idx, x_idx = divmod(index, self.nodes_per_line)
        y_idx = int(y_idx)
        x_idx = int(x_idx)

        x = self.x_min + x_idx * self.delta_x
        y = self.y_min + y_idx * self.delta_y

        if any(o.collides(x, y) for o in self.obstacles):
            return Node(id=index, x=x, y=y, neighbours=dict())

        neighbours = dict()

        # attempt to connect to all adjacent nodes
        for x_n in range(
            max(0, x_idx - 1), min(self.nodes_per_line, x_idx + 2)
        ):
            for y_n in range(max(0, y_idx - 1), min(self.no_lines, y_idx + 2)):
                if not (x_n == x_idx and y_n == y_idx):
                    xx = self.x_min + x_n * self.delta_x
                    yy = self.y_min + y_n * self.delta_y
                    if not any(o.collides(xx, yy) for o in self.obstacles):
                        idx = y_n * self.nodes_per_line + x_n
                        if idx < len(self):
                            neighbours[idx] = self.deltas[
                                (abs(x_n - x_idx), abs(y_n - y_idx))
                            ]

        return Node(id=index, x=x, y=y, neighbours=neighbours)

    def __len__(self):
        return int(self.no_lines * self.nodes_per_line)


if __name__ == '__main__':

    # a fortress
    r = Rectangle(0.5, 0.5, 1, 0.5)
    c1 = Circle(0.5, 1.0, 0.2)
    c2 = Circle(0.5, 0.5, 0.2)
    c3 = Circle(1.5, 1.0, 0.2)
    c4 = Circle(1.5, 0.5, 0.2)

    # a ridge
    r1 = Rectangle(0.0, 1.5, 1, 0.25)
    r2 = Rectangle(1.2, 1.5, 1, 0.25)

    grid = Grid(0.0, 2.0, 0.05, 0.0, 2.0, 0.05, [r, c1, c2, c3, c4, r1, r2])
    cost, sp = dijkstra(grid, 0, len(grid) - 1)
    plot_graph(grid, sp, title='Dijkstra with obstacles')
