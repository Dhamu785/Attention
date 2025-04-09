from collections import namedtuple
import warnings 

Points = namedtuple('points', ['x', 'y', 'z'])
p = Points(1, 2, 3)
print(p)
print(p.x, p.y, p.z)


from functools import partial

def add(x,y):
    warnings.warn("Trail warn")
    return x+y

added = partial(add, 5)
print(added(10))