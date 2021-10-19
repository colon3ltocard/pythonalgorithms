"""
Animates bidir A* versus bidir Dijkstra on a grid.
"""
import math
from grid_dijkstra import Grid, Rectangle, Circle, Ring
from dijkstra import dijkstra, distance, plot_graph
from bidir_dijkstra import bidir_dijkstra, make_animated_gif


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

    # a wall surrounding our starting point
    ri = Ring(0, 0, 0.2, 0.25, 0, math.pi / 8.0)
    ri2 = Ring(0.5, 0.75, 0.5, 0.6, -math.pi / 4.0, math.pi / 4.0)

    g = Grid(
        0.0, 2.0, 0.04, 0.0, 2.0, 0.04, [r, c1, c2, c3, c4, r1, r2, ri, ri2]
    )
    dst_node = g[-1]
    src_node = g[0]
    cost, sp = dijkstra(g, 0, len(g) - 1)   # our reference

    cost1, sp1, (f1, b1) = bidir_dijkstra(g, 0, len(g) - 1)

    # A*, with consistent heuristic
    cost2, sp2, (f2, b2) = bidir_dijkstra(
        g,
        0,
        len(g) - 1,
        hff=lambda x: distance(x, dst_node),
        hfb=lambda x: distance(x, src_node),
    )

    # A*, with inconsistent heuristic
    # not shortest path but a path is found quickly
    cost3, sp3, (f3, b3) = bidir_dijkstra(
        g,
        0,
        len(g) - 1,
        hff=lambda x: 6.0 * distance(x, dst_node),
        hfb=lambda x: 6.0 * distance(x, src_node),
        consistent=False,
    )

    # sanity checks
    print(cost, cost1, cost2, cost3)
    # plot_graph(g, sp2, filename="result.png", show_edges=False)
    # # let's build both animated gifs
    for title, dst_gif, (fs, bs, sps) in [
        (
            f'bidirectionnal Dijkstra, n={len(f1)}, d={cost1:.4f}',
            'bidir_grid.gif',
            (f1, b1, sp1),
        ),
        (
            f'bidirectionnal A*, n={len(f2)}, d={cost2:.4f}',
            'astar_grid.gif',
            (f2, b2, sp2),
        ),
        (
            f'bidir A* (inconsistent), n={len(f3)}, d={cost3:.4f}',
            'astar_suboptimal_grid.gif',
            (f3, b3, sp3),
        ),
    ]:
        make_animated_gif(
            title,
            g,
            dst_gif,
            fs,
            bs,
            sps,
            draw_edges=False,
            interval=50,
            blinking_ratio=0.1,
        )
