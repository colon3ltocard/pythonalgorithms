import sys
from collections import defaultdict
from itertools import dropwhile


def min_coins(coins, val):
    """
    finds the minimum number of each coins in list
    of possible 'coins' values
    to reach the target value v.
    We compute the best value ascending from
    one to our target value
    """
    # sort coins in descending order and eliminate negative and zeroes
    # also removes duplicates
    coins = sorted({c for c in coins if c > 0}, reverse=True)

    # handle 0 case
    if val == 0:
        return None, None

    # handle val < smallest coin case
    if val < min(coins):
        return None, None

    # {value:{coin_1: count_coin_1, coin_2: count_coin_2, ...}}
    best_stashes = dict()

    # there is no point exploring values below our smallest
    # coin value ('coins' is sorted in descending order)
    start_val = coins[-1]

    # we compute the best solution for each value
    # from 1 to our target, incrementaly.
    # solution for v is computed using
    # the best solutions for [1, v-1]
    for v in range(start_val, val + 1):
        best_score = sys.maxsize
        # we work only with 'possible' coins (e.g. <= value)
        # and take the best solution (lowest score)
        for coin in dropwhile(lambda x: x > v, coins):

            # for a given value, if a coin matches it is the best solution
            if coin == v:
                best_score = 1
                best_candidate = defaultdict(lambda: 0)
                best_candidate[coin] = 1
                best_stashes[v] = (best_score, best_candidate)
                break

            # compare the solution made adding our coin to the already
            # found best solution for v minus coin value
            # to the best solution among previous tested coins.
            vv = v - coin
            stash = best_stashes.get(vv)
            if stash:
                stash_score, stash = stash
                stash_score += 1
                if stash_score < best_score:
                    best_candidate = stash.copy()
                    best_candidate[coin] += 1
                    best_score = stash_score
                    best_stashes[v] = (stash_score, best_candidate)

    best_stash = best_stashes.get(val)

    if best_stash:
        final_score, best_stash = best_stash
        assert sum(k * v for k, v in best_stash.items()) == val
        return best_stash, final_score

    return None, None


if __name__ == '__main__':

    for coins, target in (
        (range(4, 102, 3), 56),
        ((1, 5, 9), 96),
        ((5, 9, 500), 1001),
        ((5, 9, 250, 500), 9786),
        ((5, 9, 50), 47),
        ((11, 49), 51),
        ((11, 49), 1),
    ):
        best_stash, best_score = min_coins(coins, target)

        if best_stash:
            print('Found a solution:')
            print(
                f'The best combination of coins is: {dict(best_stash)} for a total number of {best_score} coins and value: {sum(k*v for k,v in best_stash.items())}'
            )
        else:
            print(':-( ' * 25)
            print(
                f'no solution found with coins {coins} to reach target {target}'
            )
