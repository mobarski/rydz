from rydz import tmap, tmap_unordered


def test_tmap_sequential():
    def f(x):
        return x * 2
    assert list(tmap(f, [1, 2, 3, 4, 5], workers=1)) == [2, 4, 6, 8, 10]


def test_tmap_parallel():
    def f(x):
        return x * 2
    assert list(tmap(f, [1, 2, 3, 4, 5], workers=2)) == [2, 4, 6, 8, 10]


def test_tmap_unordered_sequential():
    def f(x):
        return x * 2
    assert sorted(tmap_unordered(f, [1, 2, 3, 4, 5], workers=1)) == [2, 4, 6, 8, 10]


def test_tmap_unordered_parallel():
    def f(x):
        return x * 2
    assert sorted(tmap_unordered(f, [1, 2, 3, 4, 5], workers=2)) == [2, 4, 6, 8, 10]
