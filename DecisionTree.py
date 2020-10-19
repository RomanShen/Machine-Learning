import numpy as np

from treePlotter import createPlot


class DecisionTree:

    def __init__(self, cls_method='ID3', classes=4):
        self.cls_method = cls_method
        self.classes = classes
        if cls_method == 'ID3':
            self.gain_func = self.Gain
        elif cls_method == 'C45':
            self.gain_func = self.GainRatio
        elif cls_method == 'CART':
            self.gain_func = self.GiniIndex
        else:
            raise ValueError('cls_methods must be ID3 or CART or C45')
        self.tree = None

    def fit(self, X, y):
        D = {}
        D['X'] = X
        D['y'] = y
        A = np.arange(X.shape[1])
        aVs = {}
        for a in A:
            aVs[a] = np.unique(X[:, a])
        self.tree = self.build_tree(D, A, aVs)

    def build_tree(self, D, A, aVs):
        """
        定义图4.2的决策树学习基本算法
        """
        X = D['X']
        y = D['y']

        # 情形1
        num_classes = np.unique(y)
        if len(num_classes) == 1:
            return num_classes[0]

        flag = True
        for a in A:
            if(len(np.unique(X[:, a])) > 1):
                flag = False
                break

        # 情形2
        if flag:
            return np.argmax(np.bincount(y))

        gains = np.zeros((len(A), ))
        if self.cls_method == 'C45':
            gains = np.zeros((len(A), 2))
        for i in range(len(A)):
            gains[i] = self.gain_func(D, A[i])

        subA = None
        if self.cls_method == 'CART':
            a_best = A[np.argmin(gains)]
            subA = np.delete(A, np.argmin(gains))
        elif self.cls_method == 'ID3':
            a_best = A[np.argmax(gains)]
            subA = np.delete(A, np.argmax(gains))
        elif self.cls_method == 'C45':
            gain_mean = np.mean(gains[:, 0])
            higher_than_mean_indices = np.where(gains[:, 0] >= gain_mean)
            higher_than_mean = gains[higher_than_mean_indices, 1][0]
            index = higher_than_mean_indices[0][np.argmax(higher_than_mean)]
            a_best = A[index]
            subA = np.delete(A, index)
        tree = {a_best: {}}

        for av in aVs[a_best]:
            indices = np.where(X[:, a_best] == av)
            Dv = {}
            Dv['X'] = X[indices]
            Dv['y'] = y[indices]
            if len(Dv['y']) == 0:
                tree[a_best][av] = np.argmax(np.bincount(y))
            else:
                tree[a_best][av] = self.build_tree(Dv, subA, aVs)
        return tree

    @classmethod
    def Ent(cls, D):
        """
        定义公式4.1信息熵
        输入D：数据集
        返回：信息熵
        """
        y = D['y']
        bin_count = np.bincount(y)
        ent = 0.
        for k in range(len(bin_count)):
            p_k = bin_count[k] / len(y)
            if p_k != 0:
                ent += p_k * np.log2(p_k)
        return -ent

    @classmethod
    def Gain(cls, D, a):
        """
        定义公式4.2信息增益
        a为某个属性的index
        """
        X = D['X']
        y = D['y']
        aV = np.unique(X[:, a])
        sum = 0.
        for v in range(len(aV)):
            Dv = {}
            indices = np.where(X[:, a] == aV[v])
            Dv['X'] = X[indices]
            Dv['y'] = y[indices]
            ent = cls.Ent(Dv)
            sum += (len(Dv['y']) / len(y) * ent)
        gain = cls.Ent(D) - sum
        return gain

    @classmethod
    def GainRatio(cls, D, a):
        """
        定义公式4.3，4.4
        参数和Gain完全相同
        """
        X = D['X']
        y = D['y']
        aV = np.unique(X[:, a])
        sum = 0.
        intrinsic_value = 0.
        for v in range(len(aV)):
            Dv = {}
            indices = np.where(X[:, a] == aV[v])
            Dv['X'] = X[indices]
            Dv['y'] = y[indices]
            ent = cls.Ent(Dv)
            sum += (len(Dv['y']) / len(y) * ent)
            intrinsic_value += (len(Dv['y']) / len(y)) * \
                np.log2(len(Dv['y']) / len(y))
        gain = cls.Ent(D) - sum
        intrinsic_value = -intrinsic_value
        gain_ratio = gain / intrinsic_value
        return np.array([gain, gain_ratio])

    @classmethod
    def Gini(cls, D):
        """
        定义公式4.5 Gini
        输入D：数据集
        返回：信息熵
        """
        y = D['y']
        bin_count = np.bincount(y)
        ent = 0.
        for k in range(len(bin_count)):
            p_k = bin_count[k] / len(y)
            ent += p_k**2
        return 1 - ent

    @classmethod
    def GiniIndex(cls, D, a):
        """
        定义公式4.6 Gini指数
        参数和Gain完全相同
        """
        X = D['X']
        y = D['y']
        aV = np.unique(X[:, a])
        sum = 0.
        for v in range(len(aV)):
            Dv = {}
            indices = np.where(X[:, a] == aV[v])
            Dv['X'] = X[indices]
            Dv['y'] = y[indices]
            ent = cls.Gini(Dv)
            sum += (len(Dv['y']) / len(y) * ent)
        gain = sum
        return gain


if __name__ == '__main__':
    """
    weather: 0-sunny, 1-windy, 2-rainny
    parents: 0-yes, 1-no
    money: 0-rich, 1-poor
    decison: 0-cinema, 1-tennis, 2-stay in, 3-shopping
    """
    data = np.array([[0, 0, 0], [0, 1, 0],
                     [1, 0, 0], [2, 0, 1],
                     [2, 1, 0], [2, 0, 1],
                     [1, 1, 1], [1, 1, 0],
                     [1, 0, 0], [0, 1, 0]])
    label = np.array([0, 1, 0, 0, 2, 0, 0, 3, 0, 1])

    # ID3
    decision_tree_id3 = DecisionTree(cls_method='ID3')
    decision_tree_id3.fit(data, label)
    print(decision_tree_id3.tree)
    createPlot(decision_tree_id3.tree)

    # C45
    decision_tree_c45 = DecisionTree(cls_method='C45')
    decision_tree_c45.fit(data, label)
    print(decision_tree_c45.tree)
    createPlot(decision_tree_c45.tree)

    # CART
    decision_tree_cart = DecisionTree(cls_method='CART')
    decision_tree_cart.fit(data, label)
    print(decision_tree_cart.tree)
    createPlot(decision_tree_cart.tree)
