import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def tmap(func, iterable, workers=os.cpu_count()):
    """Threading based parallel map that preserves order."""
    if workers == 1:
        yield from (func(item) for item in iterable)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(func, iterable)


def tmap_unordered(func, iterable, workers=os.cpu_count()):
    """Threading based parallel map with arbitrary order of the results."""
    if workers == 1:
        yield from (func(item) for item in iterable)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, item) for item in iterable]
        for future in as_completed(futures):
            yield future.result()
