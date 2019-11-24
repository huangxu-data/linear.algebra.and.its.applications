#! coding=utf-8
import numpy as np
from matplotlib import pyplot as plt


def draw_lines(augmented, start=-4, 
               stop=5, step=0.1):
    """Draw lines represented by augmented matrix on 2-d plane."""
    am = np.asarray(augmented)
    xs = np.arange(start, stop, step).reshape([1, -1])
    ys = (am[:, [-1]] - am[:, [1]]*xs) / am[:, [0]]
    for y in ys:
        plt.plot(xs[0], y)
    plt.show()


def eliminate_forward(augmented, verbose=False): 
    """消元法的前向过程"""
    A = np.asarray(augmented, dtype=np.float64)
    # row number of the last row
    pivots = []
    i, j = 0, 0
    while i < A.shape[0] and j < A.shape[1]:
        # if pivot is zero, exchange rows
        if np.isclose(A[i, j], 0):
            if (i + 1) < A.shape[0]:
                max_k = i + 1 + np.argmax(np.abs(A[i+1:, i]))
            if (i + 1) >= A.shape[0] or np.isclose(A[max_k, i], 0):
                j += 1
                continue
            A[[i, max_k]] = A[[max_k, i]]
        A[i] = A[i] / A[i, j]
        if (i + 1) < A.shape[0]:
            A[i+1:, j:] = A[i+1:, j:] - A[i+1:, [j]] * A[i, j:]
        pivots.append((i, j))
        i += 1
        j += 1
        if verbose:
            print(A)
    return A, pivots


def eliminate_backward(simplified, pivots, verbose=False):
    """消元法的后向过程."""
    A = np.asarray(simplified)
    for i, j in reversed(pivots):
        A[:i, j:] = A[:i, j:] - A[[i], j:] * A[:i, [j]] 
        if verbose:
            print(A)
    return A


def solution_check(augmented, solution):
    b = augmented[:, :-1] @ solution.reshape([-1, 1])
    b = b.reshape([-1])
    return all(np.isclose(b - augmented[:, -1], np.zeros(len(b))))


def eliminate(augmented):
    from sympy import Matrix
    print(np.asarray(augmented))
    A, pivots = eliminate_forward(augmented)
    print(" The echelon form is\n", A)
    print(" The pivots are: ", pivots)
    pivot_cols = {p[1] for p in pivots}
    simplified = eliminate_backward(A, pivots)
    if (A.shape[1]-1) in pivot_cols: # 最后一列是主元列
        print(" There is controdictory.\n", simplified)
    elif len(pivots) == (A.shape[1] -1): # 唯一解
        is_correct = solution_check(np.asarray(augmented), 
                            simplified[:, -1])
        print(" Is the solution correct? ", is_correct)
        print(" Solution: \n", simplified)
    else: # 有自由变量
        print(" There are free variables.\n", simplified)
    print("-"*30)
    print("对比Sympy的rref结果")
    print(Matrix(augmented).rref(simplify=True))
    print("-"*30)


class Vector:
    """用Python实现一个向量类，不使用Numpy."""
    def __init__(self, data:(list, tuple)):
        """data应该是一个列表或元组"""
        self._data = tuple(data)
    
    def __getitem__(self, index):
        """使下标语法和迭代生效：即x[i]有效."""
        return self._data[index]
    
    def __radd__(self, other):
        """标量或向量other与向量self相加，等价于self+other."""
        return self + other
    
    def __add__(self, other):
        """
        加法，返回元素相加后的Vector.
        
        若other是Vector，list或tuple，返回元素相加的结果；
        若other 是标量，则扩展为值全为other的向量。
        与书不一致，与Numpy一致。
        """ 
        if isinstance(other, (float, int)):
            return Vector([other + i for i in self])
        if isinstance(other, (Vector, list, tuple)):
            return Vector(left + right for left, right 
                         in zip(self, other))
        raise ValueError("the parameter is in wrong type")
    
    def __sub__(self, other):
        """
        减法，返回元素相加后的Vector.
        
        若other是Vector，list或tuple，返回元素相减的结果；
        若other 是标量，则扩展为值全为other的向量。
        与书不一致，与Numpy一致。
        """ 
        if isinstance(other, (float, int)):
            return Vector(i - other for i in self)
        if isinstance(other, (Vector, list, tuple)):
            return Vector(i - j for i, j 
                          in zip(self, other))
        raise ValueError("the parameter is in wrong type")
        
    def __rsub__(self, other):
        """标量或列表other与向量self相减，等价于self-other."""
        return other - self
    
    def __mul__(self, other):
        """
        乘法，返回元素相乘后的Vector.
        
        若other 是标量，则为标量乘向量。
        若other是Vector，list或tuple，返回元素相乘的结果；
        与书不一致，与Numpy一致。
        """ 
        if isinstance(other, (float, int)):
            return Vector(other * i for i in self)
        if isinstance(other, (Vector, list, tuple)):
            return Vector(i * j for i, j in zip(self, other))
        raise ValueError("the parameter is in wrong type")
        
    def __rmul__(self, other):
        return self * other 
    
    def __matmul__(self, other):
        if isinstance(other, (Vector, list, tuple)):
            return sum(i * j for i, j in zip(self, other))
        raise ValueError("the parameter is in wrong type")
    
    def __rmatmul__(self, other):
        return self @ other

    def __repr__(self):
        return "[" + "\n ".join(str(x) for x in self._data) + "]"


def draw_figure1():
    draw_lines([[1, -2, -1], 
                [-1, 3, 3]])
    draw_lines([[1, -2, -1], 
                [-1, 2, 3]])
    draw_lines([[1, -2, -1], 
                [-1, 2, 1]])


def section_01():
    # draw_figure1()
    # 1.1 example 1
    aug_1_1_1 = [[1, -2, 1, 0], 
                 [0, 2, -8, 8], 
                 [5, 0, -5, 10]]
    eliminate(aug_1_1_1)
    # 1.1 example 3
    aug_1_1_3 = [[0, 1, -4, 8],
                 [2, -3, 2, 1],
                 [4, -8, 12, 1]]
    eliminate(aug_1_1_3)
    eliminate([[1, -6, 4, 0, -1],
               [0, 2, -7, 0, 4],
               [0, 0, 1, 2, -3],
               [0, 0, 3, 1, 6]])
    eliminate([[0, -3, -6, 4, 9],
               [-1, -2, -1, 3, 1],
               [-2, -3, 0, 3, -1],
               [1, 4, 5, -9, -7]])
    
    eliminate([[0, 3, -6, 6, 4, -5],
               [3, -7, 8, -5, 8, 9],
               [3, -9, 12, -9, 6, 15]])
    


if __name__ == "__main__":
    section_01()