import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

class fit_distribution():
    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def Verify_H(self):
        n = len(self.data_list)
        if n == 1:
            return 0.975
        elif n == 2:
            return 0.842
        elif n == 3:
            return 0.708
        elif n == 4:
            return 0.624
        elif n == 5:
            return 0.565
        elif n == 6:
            return 0.521
        elif n == 7:
            return 0.486
        elif n == 8:
            return 0.457
        elif n == 9:
            return 0.432
        elif n == 10:
            return 0.410
        elif n == 11:
            return 0.391
        elif n == 12:
            return 0.375
        elif n == 13:
            return 0.361
        elif n == 14:
            return 0.349
        elif n == 15:
            return 0.338
        elif n == 16:
            return 0.328
        elif n == 17:
            return 0.318
        elif n == 18:
            return 0.309
        elif n == 19:
            return 0.301
        elif 20 <= n < 25:
            return 0.294
        elif 25 <= n < 30:
            return 0.27
        elif 30 <= n < 35:
            return 0.24
        elif n == 35:
            return 0.23
        else:
            return 1.36/np.sqrt(n)

    def choose_f(self, plot_switch):
        x = np.arange(0, 1, 0.001)
        # dists = {'norm': stats.norm, 'lognorm': stats.lognorm, 'expon': stats.expon}
        dists = {'norm': stats.norm}
        reasonable = self.Verify_H()
        aim_f = None
        for d in dists:
            paras = dists[d].fit(self.data_list)
            test = stats.kstest(self.data_list, dists[d].cdf, paras)
            if test[-1] > reasonable:
                # reasonable = test[-1]
                aim_f = [d, paras[-2], paras, test[-1]]
                #if test[-1] > 0.98:
                    #y = dists[aim_f[0]].pdf(x, paras[0], paras[1])
                    #plt.plot(x, y)
                    #plt.savefig('possible.png')
                    #plt.show()
        if plot_switch and aim_f is not None:
            # y = dists[aim_f[0]].pdf(x, aim_f[2][0], aim_f[2][1], aim_f[2][2])
            y = dists[aim_f[0]].pdf(x, aim_f[2][0], aim_f[2][1])
            plt.plot(x, y)
            plt.show()
        else:
            pass
        return aim_f
