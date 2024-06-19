from typing import Tuple,Iterable,Dict
import math
import heapq

class Node:
    """蒙特卡洛树节点"""
    def __init__(self,prior_prob:float,exploration:float=5,parent=None) -> None:
        """初始化节点 每个节点都有QUN 三个变量来计算选择的分数

        Args:
            `prior_prob` (float): 先验概率 `P(s, a)` 可以认为是
            `exploration` (float, optional): 探索常数. Defaults to 5.
            `parent` (_type_, optional): 父节点. Defaults to None.
        """
        self.Q = 0
        self.U = 0
        self.N = 0
        self.score = 0
        self.exploration = exploration
        self.P = prior_prob
        self.parent = parent
        self.children = {}   # type:Dict[(int) , Node]
        
    def select(self):
        """选择最大的两个子节点或者一个节点

        Returns:
            list or tuple: 如果节点是根节点的话 就直接返回一个
        """
        if self.parent is None:
            return max(self.children.items(), key=lambda item: item[1].GetScore()) #根据UCT算法计算得到 节点的得分
        else:   #如果 不是根节点 就返回两个动作
            return heapq.nlargest(2, self.children.items(), key=lambda item: item[1].GetScore()) 
    
    def Expand(self,ActionProb: Iterable[Tuple[int,float]]):
        """拓展节点

        Args:
            `ActionProb` (Iterable[Tuple[int,float]]): 一个迭代器,迭代器每次返回一个元组
            元组(动作，概率),ActionProb的长度是当前棋盘的可用落点的总数。
        """
        for action,prob in ActionProb:
            self.children[action] = Node(prior_prob=prob,parent=self)   #添加[动作,节点]对应表
    
    def __update(self,value:float):
        """更新节点的访问次数 `N(s,a)`，节点的累计奖赏`Q(s,a)`"""
        # self.Q代表 在子节点上累计的奖励
        self.Q = (self.N * self.Q + value) /(self.N + 1)
        self.N += 1 
    
    def backup(self,value:float):
        """反向传播"""
        if self.parent:
            # self.parent.backup(-value)
            self.parent.backup(-value) #TODO 
        self.__update(value)

    def GetScore(self):
        """计算节点得分"""
        self.U = self.exploration*self.P*math.sqrt(self.parent.N)/(1 + self.N) #TODO
        self.score = self.U + self.Q
        return self.score
    
    def IsLeafNode(self):
        """是否是叶节点"""
        return len(self.children) == 0