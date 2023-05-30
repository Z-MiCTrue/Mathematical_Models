import matplotlib.pyplot as plt

class points_plot():
    def __init__(self, x, y, tem_imp_col):  # 此时tem_imp_col(各点重要程度)已经归一化
        self.x = x
        self.y = y
        self.p_size = 10
        self.p_marker = 's'  # s,方; o, 圆;
        cm = plt.get_cmap("Greys")
        self.imp_col = []
        for i in tem_imp_col:
            j = cm(i)
            self.imp_col.append(j)
    
    def p_plot(self, save_switch):
        fig = plt.figure()
        fig.scatter(self.x, self.y, s = self.p_size, c = self.imp_col, marker = self.p_marker)
        if save_switch:
            plt.savefig('points_image.png')
        plt.show
        