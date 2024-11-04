import sys
import pathlib

a = pathlib.Path(sys.argv[1])
b = pathlib.Path(sys.argv[2])
assert a.exists()
assert b.exists()

a_tokens = a.glob("*.tokens")
b_tokens = b.glob("*.tokens")

