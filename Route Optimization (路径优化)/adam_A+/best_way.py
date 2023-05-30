from my_Adam import Optimizer_adam
import numpy as np
import copy
import cv2


# Calculate historical distance cost
def g_func(points):
    d = np.linalg.norm(points, axis=1)  # L2 norm
    return d


class A_search:
    def __init__(self):
        self.map = None
        self.start_point = None
        self.end_point = None
        self.best_way = []
        self.open_list = []
        self.open_list_ = []
        self.close_list = []
        self.close_list_ = []
        self.tem_g = 0
        self.top_risk = 180
        self.Oa = Optimizer_adam(2)

    # heuristic function
    def h_func(self, point_a, point_b):
        cut_num = 5
        point_T = np.linspace(point_a, point_b, cut_num + 2, dtype=int)[1: -1]  # Take 5 sample points
        # Perform adam optimization (gradient descent) on each point to approximate the real distance as possible
        for i, element in enumerate(point_T):
            self.Oa.vector_x = element
            point_T[i] = self.Oa.run()
        d_1 = np.linalg.norm(point_a - point_T[0])
        point_T_ = np.array(list(point_T[1:]) + [point_b])
        d_2 = np.sum(np.linalg.norm(point_T_ - point_T, axis=1))
        return d_1 + d_2

    def search(self):
        # Initialize
        self.Oa.variable_range = np.array([[0, self.map.shape[0]],
                                           [0, self.map.shape[1]]])
        self.open_list.append(list(self.start_point))
        self.open_list_.append([self.start_point, None, None, None])
        step_list = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]])
        step_g = g_func(step_list)
        # start iterating
        while (list(self.end_point)) not in self.open_list:
            for i, step in enumerate(step_list):
                point_test = self.open_list_[0][0] + step
                if list(point_test) in self.close_list or \
                        self.map[point_test[0], point_test[1]] <= self.top_risk:
                    pass
                elif list(point_test) in self.open_list:
                    tem_index = None
                    for i_, element in enumerate(self.open_list_):
                        if (element[0] == point_test).all():
                            tem_index = i_
                            break
                    temG_now = self.tem_g + step_g[i]
                    if temG_now < self.open_list_[tem_index][2]:
                        self.open_list_[tem_index][2] = temG_now
                        self.open_list_[tem_index][3] = copy.deepcopy(self.open_list_[0][0])
                else:
                    price = step_g[i] + self.tem_g + self.h_func(point_test, self.end_point)
                    self.open_list.append(list(point_test))
                    self.open_list_.append([point_test, price, step_g[i]+self.tem_g,
                                            copy.deepcopy(self.open_list_[0][0])])
            self.close_list.append(list(self.open_list_[0][0]))
            self.close_list_.append(copy.deepcopy(self.open_list_[0]))
            del self.open_list[self.open_list.index(list(self.open_list_[0][0]))]
            del self.open_list_[0]
            self.open_list_ = sorted(self.open_list_, key=(lambda x: x[1]), reverse=False)  # Filter the best as a node
            self.tem_g = self.open_list_[0][2]
        # backtracking path
        self.best_way.append(self.end_point)
        self.best_way.append(self.close_list_[-1][0])
        pointer = self.close_list_[-1][-1]
        while pointer is not None:
            self.best_way.append(pointer)
            for i_, element in enumerate(self.close_list_):
                if (element[0] == pointer).all():
                    pointer = element[-1]
                    break


if __name__ == '__main__':
    test_001 = A_search()
    test_001.map = cv2.imread('test.jpg', 0)
    test_001.start_point = np.array([3, 3])
    test_001.end_point = np.array([27, 27])
    test_001.search()
    for point in test_001.best_way:
        test_001.map[point[0], point[1]] = 128
    cv2.imwrite('result_map.png', test_001.map)
    cv2.imshow('', test_001.map)
    cv2.waitKey()

'''
developed by Z
'''
