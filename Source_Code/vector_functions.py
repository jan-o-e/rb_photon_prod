import numpy as np

def vec(x,y,z):
    arr=np.array([x,y,z])
    return 1/np.linalg.norm(arr)*arr

def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])