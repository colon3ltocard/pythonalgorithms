"""
Generates a valid SUDOKU
and animates step by step
using python 'rich'package.
"""
import random
from itertools import chain
import numpy as np
import time
from contextlib import contextmanager
from rich.console import Console
from rich.table import Table
from rich.align import Align
from rich.live import Live


digits = set(range(1, 10))


def generate_sudoku() -> np.ndarray:
    """
    generates a valid sudoku board of
    9*9 cells
    """

    arr = np.zeros((9, 9), dtype=np.uint8)
    moves = []

    def try_value(ti: int, tj: int, val: np.uint8) -> bool:
        """
        Recursive function to populate the sudoku.
        """
        arr[ti, tj] = val
        moves.append((ti, tj, val))
        if len(arr[arr != 0]) == arr.shape[0] * arr.shape[1]:
            return True

        else:

            i, j = np.where(arr == 0)
            i, j = i[0], j[0]
            si, sj = (i // 3) * 3, (j // 3) * 3
            candidates = list(
                digits.difference(
                    {
                        x
                        for x in chain(
                            arr[i, :],
                            arr[:, j],
                            arr[si : si + 3, sj : sj + 3].flat,
                        )
                    }
                )
            )
            random.shuffle(candidates)
            for v in candidates:
                r = try_value(i, j, v)
                if r:
                    return r

            arr[ti, tj] = 0
            moves.append((ti, tj, 0))
            return False

    r = try_value(
        random.randint(0, 8), random.randint(0, 8), random.choice(list(digits))
    )
    if r:
        return arr, moves
    else:
        raise Exception('failed to generate a Sudoku ?!?!')


if __name__ == '__main__':
    arr, moves = generate_sudoku()
    console = Console()

    def generate_table(nparr):
        table = Table()
        table = Table(show_footer=False, show_lines=True, show_header=False)
        table_centered = Align.center(table)
        for i in range(9):
            table.add_row(*[str(x) if x != 0 else ' ' for x in nparr[:, i]])
        return table_centered

    console.clear()
    BEAT_TIME = 0.04

    @contextmanager
    def beat(length: int = 1) -> None:
        yield
        time.sleep(length * BEAT_TIME)

    display_arr = np.zeros((9, 9), np.uint8)
    with Live(
        generate_table(display_arr),
        console=console,
        screen=False,
        refresh_per_second=20,
    ) as live:
        for i, j, v in moves:
            with beat(4):
                display_arr[i, j] = v
                live.update(generate_table(display_arr))
