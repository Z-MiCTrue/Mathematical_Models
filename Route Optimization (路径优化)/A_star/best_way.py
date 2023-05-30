import numpy as np
import copy
import cv2


def h_func(point_a, point_b):
    # d = np.max(np.abs(point_a - point_b))  # 切比雪夫
    d = np.linalg.norm(point_a - point_b)  # 欧几里得
    return d


def g_func(points):
    d = np.linalg.norm(points, axis=1)  # 范数
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
        self.top_risk = 200

    def re_init(self):
        self.best_way = []
        self.open_list = []
        self.open_list_ = []
        self.close_list = []
        self.close_list_ = []
        self.tem_g = 0

    def search(self):
        self.open_list.append(list(self.start_point))
        self.open_list_.append([self.start_point, None, None, None])
        step_list = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]])
        step_g = g_func(step_list)
        while (list(self.end_point)) not in self.open_list:
            for i, step in enumerate(step_list):
                point_test = self.open_list_[0][0] + step
                if (point_test < 0).any() or (point_test >= self.map.shape).any():
                    pass
                else:
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
                        price = step_g[i] + self.tem_g + h_func(point_test, self.end_point)
                        self.open_list.append(list(point_test))
                        self.open_list_.append([point_test, price, step_g[i] + self.tem_g,
                                                copy.deepcopy(self.open_list_[0][0])])
            self.close_list.append(list(self.open_list_[0][0]))
            self.close_list_.append(copy.deepcopy(self.open_list_[0]))
            del self.open_list[self.open_list.index(list(self.open_list_[0][0]))]
            del self.open_list_[0]
            self.open_list_ = sorted(self.open_list_, key=(lambda x: x[1]), reverse=False)  # False: 升序
            self.tem_g = self.open_list_[0][2]
        self.best_way.append(self.end_point)
        self.best_way.append(self.close_list_[-1][0])
        pointer = self.close_list_[-1][-1]
        while pointer is not None:
            self.best_way.append(pointer)
            for i_, element in enumerate(self.close_list_):
                if (element[0] == pointer).all():
                    pointer = element[-1]
                    break
            pass


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
