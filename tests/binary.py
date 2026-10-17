from seamless import Buffer
import numpy as np
from io import BytesIO

arr = np.arange(20)
b = BytesIO()
np.save(b, arr)
buf1 = Buffer(b.getvalue())

cs1 = buf1.get_checksum()

buf2 = Buffer(arr, "binary")
cs2 = buf2.get_checksum()

assert cs1 == cs2, (cs1, cs2)

arr2 = buf2.get_value("binary")  # read-only copy
print(arr2[:4])
original = arr2.copy()
try:
    arr2[:2] = 999  # expected to raise ValueError
except ValueError:
    pass
else:
    raise AssertionError("read-only array accepted assignment")
assert np.array_equal(arr2, original)
