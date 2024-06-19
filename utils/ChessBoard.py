from collections  import OrderedDict
import copy
from typing import Tuple
import torch
import numpy as np

class ChessBoard:
    
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    
    def __init__(self,boardlen= 17,FeaturePlaneCnt=7) -> None:
        """
        Args:
            `boardlen` (int, optional): 棋盘的边长. Defaults to 17.
            `FeaturePlaneCnt` (int, optional): 特征平面个数. Defaults to 7.
        """
        self.boardlen = boardlen
        self.CurrentPlayer = self.BLACK
        self.FeaturePlaneCnt = FeaturePlaneCnt
        self.AvailableACtions = list(range(boardlen**2))
        # 棋盘中落的子用字典存储,key为action,value为CurrentPlayer
        self.state = OrderedDict()  # type: OrderedDict[list,int]
        self.boardstate = OrderedDict() # type: OrderedDict[int,int]
        self.PreviousAction = None
    
    def copy(self):
        """深拷贝复制棋盘"""
        return copy.deepcopy(self)

    def ClearBoard(self):
        """清空棋盘状态，初始化相关参数"""
        self.state.clear()
        self.boardstate.clear()
        self.PreviousAction = None
        self.CurrentPlayer = self.BLACK
        self.AvailableACtions = list(range(self.boardlen**2))
        
    
    def DoAction(self,action: list):
        """模拟执行动作
        Args:
            `action`: list, 一个存储int数字的列表
        
        """
        self.PreviousAction = action    #记录最新动作为action
        try:
            for i in action:
                if i in self.AvailableACtions:
                    self.AvailableACtions.remove(i) # 可选动作去除当前执行的动作
                self.boardstate[i] = self.CurrentPlayer # boardstate 储存每个独立棋子位置的玩家信息
        except TypeError:
            print("action is ",action)
            print("action type is ",type(action))
        self.state[tuple(action)] = self.CurrentPlayer  # 字典当中加入玩家动作
        self.CurrentPlayer = 3 - self.CurrentPlayer #切换角色 
        
    def IsGameOver(self) -> Tuple [bool,int]:
        """ 判断游戏是否结束(只针对上一步去判断，最新的一步之前的所有步骤都没有判断出来)

        Returns
        -------
        IsOver: bool
            游戏是否结束，分出胜负或者平局则为 `True`, 否则为 `False`

        Winner: int
            游戏赢家，有以下几种:
            * 如果游戏分出胜负，则为 `ChessBoard.BLACK` 或 `ChessBoard.WHITE`
            * 如果还有分出胜负或者平局，则为 `None`
        """
        if len(self.state) < 6: #如果当前局面上所有的步数不足6个，就直接判断游戏还没结束
            return False,None
        """
棋子数变化
  黑    1 1 3 3 5 7
  白    0 2 2 4 4 6
        """
        
        n = self.boardlen
        acts = self.PreviousAction
        
        player = self.state[tuple(acts)]
        
        directions = [[(0, -1),  (0, 1)],   # 水平搜索
                      [(-1, 0),  (1, 0)],   # 竖直搜索
                      [(-1, -1), (1, 1)],   # 主对角线搜索
                      [(1, -1),  (-1, 1)]]  # 副对角线搜索
        #搜索的时候从最新的两个点出发，如果这两个点的中心能构成一个连6子,那就说明游戏结束
        #否则的话就说明
        for act in acts:
            row, col = act//n, act%n # row,col 的范围是[0,boardlen-1]
            for i in range(4):
                cnt = 0 #cnt 用来记录这一直线的棋子连起来的个数
                for j in range(2):
                    x = row + directions[i][j][0]
                    y = col + directions[i][j][1]
                    if x >= 0 and x < n and y >= 0 and y < n and x*n+y in self.boardstate and self.boardstate[x*n+y] == player:
                        cnt += 1
                    else: 
                        break
                if cnt>=6:
                    return True,player
        # 平局,当前已经没有可以下的棋子位置了
        if not self.AvailableACtions:
            return True,None
        
        return False,None   #否则就是对局没有结束，没有玩家胜出
                    
    def GetFeaturePlanes(self) -> torch.Tensor:
        """得到棋盘状态特征，维度是(FeaturePlanesCnt,boardlen,boardlen)

        Returns:
        `FeaturePlanes`: Tensor of shape `(n_feature_planes, board_len, board_len)`
            特征平面图像
        """
        n = self.boardlen
        FeaturePlanes = torch.zeros((self.FeaturePlaneCnt,n**2))
        
        if self.state:
            actions = np.array(list((self.state.keys())))[::-1] #之前的所有动作从队列当中取出 并反转使得最新的排列在最前面
            players = np.array(list(self.state.values()))[::-1] #之前的所有动作从玩家信息当中取出 并反转使得最新的排列在最前面
            xactions = actions[players == self.CurrentPlayer]
            yactions = actions[players != self.CurrentPlayer]
            for i in range(self.FeaturePlaneCnt):
                if i%2 == 0:    #如果i是偶数
                    for j in range(i,len(xactions)):    #TODO 这里可能会出现问题
                        # for index in xactions[j]:
                        #     FeaturePlanes[i][xactions[j][index]] = 1
                        # FeaturePlanes[i][xactions[j][:]] = 1
                        # print("j is ",j)
                        FeaturePlanes[i,xactions[j]] = 1
                else:
                    for j in range(i,len(yactions)):
                        # for index in yactions[j]:
                        #     FeaturePlanes[i][yactions[j][index]] = 1
                        # FeaturePlanes[i][yactions[j][:]] = 1
                        FeaturePlanes[i,yactions[j]] = 1
                        
        return FeaturePlanes.view(self.FeaturePlaneCnt,n,n)
            
            
    
        
        