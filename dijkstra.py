"""
Implementation of Dijkstra shortest path
algorithm in python.
See https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm.
"""
import sys
import math
from collections import defaultdict, namedtuple
import random
from heapq import heappush, heappop
from matplotlib import pyplot as plt


Node = namedtuple('Node', ['x', 'y', 'id', 'neighbours'])


def distance(a: Node, b: Node) -> float:
    """
    returns squared euclidean distance between nodes a and b
    """
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)


def connect_nodes(a: Node, b: Node):
    """
    Connects the nodes a and b.
    """
    d = distance(a, b)
    a.neighbours[b.id] = d
    b.neighbours[a.id] = d


def generate_random_graph(
    no_nodes: int, connect_probability: float = 0.5
) -> list[Node]:
    """
    Generates a graph of nodes in the euclidean plane walking
    from one node to the next and randomly connecting existing nodes.
    """
    # sanity check on connect probability
    connect_probability = max(0.0, min(abs(connect_probability), 1.0))
    delta = 1.0 / no_nodes
    sigma = 0.1
    nodes: list[Node] = list()
    while len(nodes) < no_nodes:
        node_id = len(nodes)

        if node_id == 0:
            # starts lower left corner of our [0, 1]; [0, 1] bbox
            x = random.uniform(0.0, 0.05)
            y = random.uniform(0.0, 0.05)
        else:
            x = random.gauss(min(node_id * delta, 1.0), sigma)
            y = random.gauss(min(node_id * delta, 1.0), sigma)

        node = Node(
            x=x,
            y=y,
            id=node_id,
            neighbours=dict(),
        )
        if node_id > 0:
            connect_nodes(node, nodes[-1])

        if (
            1 < node_id < no_nodes - 1
            and random.uniform(0, 1.0) <= connect_probability
        ):
            neighbour = random.choice(nodes[:-1])
            connect_nodes(node, neighbour)

        nodes.append(node)

    return nodes


def plot_graph(
    nodes: list[Node],
    shortest: list[int] = None,
    title: str = '',
    filename='imgs/results.png',
    show_edges=True,
):
    """
    plots the graph using matplotlib
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    if show_edges:
        edges = {tuple(sorted((n.id, x))) for n in nodes for x in n.neighbours}
        for edge in edges:
            from_node, to_node = [nodes[x] for x in edge]
            x = [n.x for n in (from_node, to_node)]
            y = [n.y for n in (from_node, to_node)]
            ax.plot(x, y, color='gray', linewidth=0.5)

    x, y = [n.x for n in nodes if n.neighbours], [
        n.y for n in nodes if n.neighbours
    ]
    ax.scatter(x, y, color='gray', s=10)

    if shortest:
        nodes_in_path = [nodes[idx] for idx in shortest]
        x = [n.x for n in nodes_in_path]
        y = [n.y for n in nodes_in_path]
        ax.plot(x, y, color='green', linewidth=1)

        start_node = nodes[shortest[0]]
        ax.scatter(start_node.x, start_node.y, color='green', s=40)

        on_path_nodes = [nodes[ids] for ids in shortest[1:-1]]
        ax.scatter(
            [n.x for n in on_path_nodes],
            [n.y for n in on_path_nodes],
            c=range(len(on_path_nodes)),
        )

        end_node = nodes[shortest[-1]]
        ax.scatter(end_node.x, end_node.y, color='red', s=40)
    ax.set_aspect('equal')
    plt.savefig(filename, dpi=300)


def build_shortest_path(previous: dict, index: int, target: int) -> list[int]:
    """
    builds the shortest path
    traversing our previous dict
    """
    res = [index, previous[index]]
    index = previous[index]

    while index != target:
        index = previous[index]
        res.append(index)
    return res[::-1]


def dijkstra(nodes: list[Node], src_id: int, dst_id: int) -> list[int]:
    """
    Computes the shortest path from source node to dst node
    using Dijkstra algorithm.
    """
    visited_nodes = set()
    h: list[tuple[float, Node]] = []
    previous = dict()
    distances = defaultdict(lambda: sys.maxsize)
    distances[src_id] = 0.0

    heappush(h, (0.0, nodes[src_id]))

    while h:
        _, node = heappop(h)

        if node.id in visited_nodes:
            continue

        dist = distances[node.id]

        if node.id == dst_id:
            return distances[dst_id], list(
                build_shortest_path(previous, dst_id, src_id)
            )

        for n, d in (
            (nodes[k], v)
            for k, v in node.neighbours.items()
            if k not in visited_nodes
        ):
            new_dist = dist + d
            if new_dist <= distances[n.id]:
                distances[n.id] = new_dist
                previous[n.id] = node.id

                heappush(h, (new_dist, n))

        visited_nodes.add(node.id)

    return []


def validate_solution(nodes: list[Node], path: list[int]):
    """
    Given a list of nodes and the list of indexes
    of the shortest path, validates that the path exits
    by walking from one node to the next.
    """
    for idx, node in enumerate([nodes[k] for k in path[:-1]]):
        if path[idx + 1] not in node.neighbours:
            raise Exception(
                f'next node {path[idx+1]} not in {node} neighbours'
            )


if __name__ == '__main__':
    g = generate_random_graph(1000, connect_probability=0.1)

    src = g[0]
    dst = g[-1]
    c, d = dijkstra(g, src.id, dst.id)
    validate_solution(g, d)
    print(
        f"shortest path from {src.id} to {dst.id} is {' -> '.join(str(x) for x in d)} with distance {c}"
    )
    plot_graph(g, d)
