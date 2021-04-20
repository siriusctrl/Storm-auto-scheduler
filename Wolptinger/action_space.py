import pyflann
from gym.spaces import box
import numpy as np
import itertools

class Space:

    def __init__(self, low, high, points):
        """
        The Space class here trying to discretise the continuous action space,
        and then apply the Wolptinger algorithm.
        """
        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        self._space = init_uniform_space([0] * self._dimensions,
                                          [1] * self._dimensions,
                                          points)
        # print("self._space: {}, self._space.shape: {}, self._space.dtype: {}".format(self._space, self._space.shape, self._space.dtype))
        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self._space, algorithm='kdtree')
        # print("Index type: {}".format(type(self._index)))
    
    def search_point(self, point, k):
        p_in = self.import_point(point).reshape(1, -1).astype('float64')
        # print("p_in: {}, p_in.shape: {}, p_in.dtype: {}".format(p_in, p_in.shape, p_in.dtype))
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self._space[search_res]
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]
        return np.array(p_out)

    def import_point(self, point):
        return (point - self._low) / self._range
    
    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self._space

    def shape(self):
        return self._space.shape

    def get_number_of_actions(self):
        return self.shape()[0]



class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        0, 1, 2, ..., n-2, n-1
    """

    def __init__(self, n):  # n: the number of the discrete actions
        super().__init__([0], [n - 1], n)

    def export_point(self, point):
        return super().export_point(point).astype(int)


class Binary_Discrete_space(Space):
    """
        Discrete action space with n dimensions binary values
    """

    def __init__(self, dim, space=None):
        super().__init__([0]*dim, [1]*dim, 2**dim)
        
        if space:
            self._space = space

    def export_point(self, point):
        return super().export_point(point).astype(int)


def init_uniform_space(low, high, points):
    dims = len(low)
    points_in_each_axis = round(points**(1 / dims))

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    return np.array(space)

if __name__ == '__main__':
    # print(init_uniform_space([0]*5, [1]*5, 2**5))
    b = Binary_Discrete_space(5)
    # print(b.import_point([0,0,0,0,1]))
    print(b.export_point(np.array([0., 0., 0., 0., 1.])))