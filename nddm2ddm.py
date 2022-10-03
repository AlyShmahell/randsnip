import numpy as np

def nddm2ddm(mat):
    """
    makes a copy of a numpy array 
    then converts said copy to a 
    diagonally dominant array if possible
    """
    n, m    = mat.shape
    if n != m: return np.zeros((0,))
    maxrow  = np.amax(mat, axis=-1)
    restrow = np.sum(mat, axis=-1) - maxrow
    if any(maxrow < restrow): return np.zeros((0,))
    maxind  = np.argmax(mat, axis=-1)
    res     = mat.copy()
    res[maxind, :] = mat
    return res

if __name__ == '__main__':
    mat = np.array([ 
        [4, 2, 1, 101, 1],
        [5, 104, 3, 4 ,1],
        [105, 5, 2, 2, 3],
        [5, 4, 105, 4, 4],
        [2, 4, 4, 1, 101]
    ])
    res = nddm2ddm(mat)
    print(res)