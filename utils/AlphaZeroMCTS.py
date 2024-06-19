#conding: utf-8
from .PolicyValueNet import PolicyValueNet
from .Node  import Node
from .ChessBoard import ChessBoard
import numpy as np

class AlphaZeroMCTS:
    def __init__(self,PolicyValueNet:PolicyValueNet,exploration= 4.0,IterCnt=1200,IsSelfPlay=False) -> None:
        """初始化函数

        Args:
            `PolicyValueNet` (PolicyValueNet): 策略价值网络
            `exploration` (float, optional): 探索常数. Defaults to 4.0.
            `IterCnt` (int, optional): 迭代次数. Defaults to 1200.
            `IsSelfPlay` (bool, optional): 是否是自我博弈状态. Defaults to False.
        """
        self.exploration = exploration
        self.IterCnt = IterCnt
        self.IsSelfPlay = IsSelfPlay
        self.Net = PolicyValueNet
        self.Net.set_device(False)  #TODO
        self.root = Node(prior_prob=1,parent=None)  #根节点
    
    def GetAction(self,ChessBoard:ChessBoard):
        """根据当前局面返回下一步的动作
        """
        
        for i in range(self.IterCnt):
            board = ChessBoard.copy()
            
            node = self.root
            #如果没有遇到叶节点，就一直向下搜索并更新棋盘
            while not node.IsLeafNode():
                SelectedAction = node.select()
                if isinstance(SelectedAction,list):
                    node = SelectedAction[0][1]         # 返回的是两个子节点，也就是选择了两个节点作为下一个Action,但是只能设置一个node为子节点
                    board.DoAction([SelectedAction[0][0],
                                    SelectedAction[1][0]])
                else:   # SelectedAction  is  a tuple
                    node = SelectedAction[1]
                    board.DoAction([SelectedAction[0]]) 
            
            # 好了现在的Node节点就是叶子节点
            
            # 判断游戏有没有结束
            IsOver, winner = board.IsGameOver()
            p, value = self.Net.predict(board)  
            #在经历过多次DoAction之后，到达了根节点 现在要扩展新的节点，也就是选择新的一个Action去执行
            #在游戏还没有结束的前提下，Action要选择可以通过现有的神经网络计算得到
            
            if not IsOver:
                # 添加迪利克雷噪声
                if self.IsSelfPlay:
                    p = 0.75*p + 0.25 * np.random.dirichlet(0.03*np.ones(len(p)))
                node.Expand(zip(board.AvailableACtions,p))      # 这个操作会将所有的可以选择的下一个action都计算得到一个概率值
            elif winner is None:
                value = 0
            else:
                value = 1 if winner == board.CurrentPlayer else -1
            
            node.backup(-value) # 最后的value用于更新局面
        
        #计算各个动作的概率，在自我博弈的状态下
        T = 1 if self.IsSelfPlay and len(ChessBoard.state) <= 30 else 1e-3
        visits = np.array([i.N for i in self.root.children.values()]) #visits计算根节点的每个子节点的拜访次数
        Pi_ = self.__getPi(visits,T)
        
        # 根据Pi 选出动作及其对应的节点
        actions = list(self.root.children.keys())   #TODO
        action = int(np.random.choice(actions,p=Pi_))    #根据概率Pi 选择action,这些action都是经过网络计算得到概率之后通过Expand函数拓展出来的
        # 
        
        if self.IsSelfPlay :
            # 创建一个维度为boardlen**2 的 
            Pi = np.zeros(ChessBoard.boardlen**2)
            Pi[actions] = Pi_   # 只有有对应的动作的节点才有有选择的概率
            self.root = self.root.children[action]  # 更新root,下次从这里开始搜索，
            self.root.parent = None
            return action, Pi
        else:
            self.ResetRoot()
            return action
            
    def __getPi(self,visits,T):
        """根据节点的访问次数计算Pi"""
        
        x = 1/T * np.log(visits+1e-11)
        x = np.exp(x - x.max())
        Pi = x/x.sum()
        return Pi
    
    def ResetRoot(self):
        """重置根节点"""
        self.root = Node(prior_prob=1,exploration=self.exploration,parent=None)
    
    def SetSelfPlay(self,IsSelfPlay:bool):
        """重新设置一下博弈状态"""
        self.IsSelfPlay = IsSelfPlay