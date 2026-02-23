from rydz import tmap_unordered


def test_tmap_sequential():
    def f(x):
        return x * 2
    iter = tmap_unordered(f, [1, 2, 3, 4, 5], workers=1)
    assert list(sorted(iter)) == [2, 4, 6, 8, 10]


def test_tmap_parallel():
    def f(x):
        return x * 2
    iter = tmap_unordered(f, [1, 2, 3, 4, 5], workers=2)
    assert list(sorted(iter)) == [2, 4, 6, 8, 10]
