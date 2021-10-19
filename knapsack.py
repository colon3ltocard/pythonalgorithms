"""
Generates, solves and animate
a knapsack problem
https://en.wikipedia.org/wiki/Knapsack_problem
"""
import random
from operator import attrgetter
from dataclasses import dataclass
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import get_cmap


@dataclass
class Item:
    weight: int
    value: float

    def __post_init__(self):
        # we add a relative value attribute
        # to facilitate sorting and greedy algo
        self.relval = self.value / self.weight


@dataclass
class KnapSack:
    weight: int
    items: list[Item]

    def upper_bound(self, item_index: int, candidates: list[Item]) -> float:
        """
        computes upper bound for current knapsack value
        adding items from item_index included.
        THe last item added, if exceeding allowed weight,
        will be added partially. This is the trick to have a upper bound
        since items are sorted by decreasing relative value.
        """
        w = sum(x.weight for x in self.items)
        v = sum(x.value for x in self.items)

        for c in candidates[item_index:]:
            if w + c.weight > self.weight:
                v += ((self.weight - w) / c.weight) * c.value
                break
            else:
                w += c.weight
                v += c.value
        return v

    def greedy(self, candidates: list[Item]) -> list[Item]:
        """
        greedy solution, stash items in decreasing relval
        order as long as they fit.
        """
        w = 0
        v = 0
        candidates = sorted(candidates, key=attrgetter('relval'), reverse=True)
        r = list()
        for c in candidates:
            if c.weight + w > self.weight:
                continue
            else:
                w += c.weight
                v += c.value
                r.append(c)
        return r, v

    def solve(
        self, candidates: list[Item], bruteforce: bool = False
    ) -> list[Item]:
        """
        Solves the knapsack using branch and bound.
        bruteforce boolean allows to force searching all branches
        """
        # makes sure our candidates are sorted
        candidates = sorted(candidates, key=attrgetter('relval'), reverse=True)

        # initialises our best result with a fast greedy search
        r, v = self.greedy(candidates)

        # a list of moves to record each step
        moves = list()

        niter = 0
        knap = self

        def solver(item_index: int):
            nonlocal v, r, niter, knap
            if bruteforce or knap.upper_bound(item_index, candidates) > v:
                niter += 1
                nc = candidates[item_index]

                # solve branch including this item - if possible
                if nc.weight + sum(x.weight for x in knap.items) <= knap.weight:
                    knap.items.append(nc)
                    moves.append((1, nc))
                    nv = sum(x.value for x in knap.items)
                    if nv > v:
                        v, r = nv, knap.items.copy()
                    if item_index + 1 < len(candidates):
                        solver(item_index + 1)
                    knap.items.pop()
                    moves.append((0, nc))

                # solve branch excluding this item
                if item_index + 1 < len(candidates):
                    solver(item_index + 1)

        self.items = list()
        solver(0)
        self.items = r
        return r, niter, moves


def generate_knapsack(
    no_items: int, max_weight: int
) -> tuple[KnapSack, list[Item]]:
    """
    Builds a knapsack problem
    """

    items = sorted(
        [
            Item(
                weight=random.randint(1, max_weight // 2),
                value=random.uniform(0.1, 1.0),
            )
            for _ in range(no_items)
        ],
        key=attrgetter('relval'),
        reverse=True,
    )
    knap = KnapSack(
        weight=random.randint(int(max_weight * 0.9), max_weight), items=list()
    )
    return knap, items


class KnapAnimator:
    """
    Builds an animation from
    a knapsack problem solution
    """

    def __init__(
        self, knap: KnapSack, candidates: list[Item], title=''
    ) -> None:
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)
        self.fig.suptitle(title)
        plt.tight_layout()
        self.candidates = candidates
        r, v = knap.greedy(candidates)
        random.shuffle(self.candidates)
        self.v = 0
        self.knap = knap
        self.jet = get_cmap('jet', len(candidates))

        # plot 3 holds the greedy solution
        self.ax3.clear()
        bottom = 0
        labels = [f'Value: {float(v):.5}, w: {sum(x.weight for x in r)}']
        for idx, item in enumerate(r):
            self.ax3.bar(
                labels,
                [item.weight],
                color=self.jet(self.candidates.index(item)),
                bottom=bottom,
                hatch='///' if idx%2 else '\\\\\\',
            )
            bottom = sum(x.weight for x in r[: idx + 1])
        self.ax3.axhline(y=self.knap.weight, linewidth=2, color='red')
        self.ax3.set_ylim(0, self.knap.weight * 1.05)
        self.ax3.set_ylabel('weight')
        self.ax3.set_title('Greedy solution')

    def update(self, items: list[Item]):
        """
        Updates the bar chart with each item.
        if a new best is found the second bar
        plot on the right is updated
        """
        v = sum(x.value for x in items)
        w = sum(x.weight for x in items)
        axes = [self.ax1]
        if v > self.v:
            self.v = v
            axes.append(self.ax2)
        for ax in axes:
            ax.clear()
            bottom = 0
            labels = [f'Value: {float(v):.5}, w: {w}']
            for idx, item in enumerate(items):
                ax.bar(
                    labels,
                    [item.weight],
                    color=self.jet(self.candidates.index(item)),
                    bottom=bottom,
                    hatch='///' if idx%2 else '\\\\\\',
                )
                bottom = sum(x.weight for x in items[: idx + 1])
            ax.axhline(y=self.knap.weight, linewidth=2, color='red')
            ax.set_ylim(0, self.knap.weight * 1.05)
            ax.set_ylabel('weight')
        self.ax1.set_title('Current solution')
        self.ax2.set_title('Best solution')
        plt.tight_layout()
        return axes

    def make_animated_gif(
        self,
        moves: list[tuple[int, int]],
        writer: str = 'ffmpeg',
        interval: int = 500,
        dst_file: str = 'knapsack.gif',
    ):
        """
        Makes an animated gif out of two sequences of forward (fs) and backward (bs)
        path-finding algorithm. The final shortest path will be blinked.
        """
        items = []

        def bar_gen():
            for action, item in moves:
                if action:
                    items.append(item)
                else:
                    items.pop()
                yield items

        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            bar_gen(),
            interval=interval,
            blit=False,
            repeat_delay=500,
            save_count=len(moves),
        )
        ani.save(f'imgs/{dst_file}', writer=writer)


if __name__ == '__main__':
    rgreed = r = []
    while rgreed == r:
        print("Generating new knapsack problem")
        knap, candidates = generate_knapsack(100, 500)
        rgreed, v = knap.greedy(candidates)
        r, niter, m = knap.solve(candidates)
    ca = KnapAnimator(
        knap,
        candidates,
        title=f'Knapsack of weight {knap.weight} solved in {niter} iterations',
    )
    ca.make_animated_gif(m)
    print(niter)
    print(knap.weight, r)
    print(sum(x.value for x in r), knap.weight)
