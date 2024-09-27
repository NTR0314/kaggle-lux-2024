import numpy as np

from rux_2024._lowlevel import hello_numpy_world, hello_world


def test_hello_world() -> None:
    assert hello_world() == "Hello from rux-2024!"


def test_numpy_hello() -> None:
    arr = hello_numpy_world()
    assert arr.shape == (4, 2)
    assert arr.dtype == np.float32
    assert arr[0, 0] == 1
    assert arr[3, 1] == 2
