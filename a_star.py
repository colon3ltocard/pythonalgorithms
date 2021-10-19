"""
Implementation and animation of the a* pathfinding
algorithm.
See https://en.wikipedia.org/wiki/A*_search_algorithm
"""
from dijkstra import dijkstra, generate_random_graph, distance
from bidir_dijkstra import make_animated_gif, bidir_dijkstra


if __name__ == '__main__':

    g = generate_random_graph(200, connect_probability=0.1)
    dst_node = g[-1]
    src_node = g[0]
    cost, sp = dijkstra(g, 0, len(g) - 1)   # our reference

    cost1, sp1, (f1, b1) = bidir_dijkstra(g, 0, len(g) - 1)

    # in our case, a* is only dijkstra with a non zero
    # heuristic equal to the distance to the target node.
    # hence we specify the heuristic function forward (hff)
    # and backward (hfb)
    cost2, sp2, (f2, b2) = bidir_dijkstra(
        g,
        0,
        len(g) - 1,
        hff=lambda x: distance(x, dst_node),
        hfb=lambda x: distance(x, src_node),
    )

    # let's build both animated gifs
    for title, dst_gif, (fs, bs, sps) in [
        (f'bidir Dijkstra, n={len(f1)}', 'bidir_150.gif', (f1, b1, sp1)),
        (f'bidir A*, n={len(f2)}', 'astar_150.gif', (f2, b2, sp2)),
    ]:

        make_animated_gif(title, g, dst_gif, fs, bs, sps)
