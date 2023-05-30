import numpy as np
import matplotlib.pyplot as plt


def cubic(start,end,*args):
    count=4*(len(args)-1)
    mat_ori=np.zeros((count,count))
    mat_ans=np.zeros(count)
    index=0
    for i,j in zip(args[:-2],args[1:-1]):
        mat_ori[4*index,4*index]=i[0]**3
        mat_ori[4*index,4*index+1]=i[0]**2
        mat_ori[4*index,4*index+2]=i[0]
        mat_ori[4*index,4*index+3]=1
        mat_ans[4*index]=i[1]
        mat_ori[4*index+1,4*index]=j[0]**3
        mat_ori[4*index+1,4*index+1]=j[0]**2
        mat_ori[4*index+1,4*index+2]=j[0]
        mat_ori[4*index+1,4*index+3]=1
        mat_ans[4*index+1]=j[1]
        mat_ori[4*index+2,4*index]=3*j[0]**2
        mat_ori[4*index+2,4*index+1]=2*j[0]
        mat_ori[4*index+2,4*index+2]=1
        mat_ori[4*index+2,4*index+4]=-3*j[0]**2
        mat_ori[4*index+2,4*index+5]=-2*j[0]
        mat_ori[4*index+2,4*index+6]=-1
        mat_ans[4*index+2]=0
        mat_ori[4*index+3,4*index]=6*j[0]
        mat_ori[4*index+3,4*index+1]=2
        mat_ori[4*index+3,4*index+4]=-6*j[0]
        mat_ori[4*index+3,4*index+5]=-2
        mat_ans[4*index+3]=0
        index+=1
    mat_ori[4*index,4*index]=args[-2][0]**3
    mat_ori[4*index,4*index+1]=args[-2][0]**2
    mat_ori[4*index,4*index+2]=args[-2][0]
    mat_ori[4*index,4*index+3]=1
    mat_ans[4*index]=args[-2][1]
    mat_ori[4*index+1,4*index]=args[-1][0]**3
    mat_ori[4*index+1,4*index+1]=args[-1][0]**2
    mat_ori[4*index+1,4*index+2]=args[-1][0]
    mat_ori[4*index+1,4*index+3]=1
    mat_ans[4*index+1]=args[-1][1]
    mat_ori[4*index+2,0]=3*args[0][0]**2
    mat_ori[4*index+2,1]=2*args[0][0]
    mat_ori[4*index+2,2]=1
    mat_ans[4*index+2]=start
    mat_ori[4*index+3,4*index]=3*args[-1][0]**2
    mat_ori[4*index+3,4*index+1]=2*args[-1][0]
    mat_ori[4*index+3,4*index+2]=1
    mat_ans[4*index+3]=end
    mat_rg=np.linalg.solve(mat_ori,mat_ans)
    def rtn_func(x):
        def bin_search(left,right):
            if(x<args[left][0]):return left
            if(x>args[right-1][0]):return right-1
            if(right-left<=1):return left
            mid=int((left+right)/2)
            if(x<args[mid][0]):return bin_search(left,mid)
            else:return bin_search(mid,right)
        num=bin_search(0,len(args)-1)
        return mat_rg[4*num]*x**3+mat_rg[4*num+1]*x**2+mat_rg[4*num+2]*x+mat_rg[4*num+3]
    return rtn_func


if __name__ == '__main__':
    data = [[0, 0], [1, 1]]  # 数据插值点
    func1 = cubic(1, 1, *data)
    x = np.arange(-1, 7, 0.1)
    y1 = [func1(i) for i in x]
    plt.plot(x, y1)
    plt.plot([i[0] for i in data], [i[1] for i in data], 'ob')
    plt.show()
