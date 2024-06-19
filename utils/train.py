
# from torch import nn
from torch.nn import functional as F
from torch import optim,cuda
from torch.nn import Module
import torch,json
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from collections import namedtuple
from .PolicyValueNet import PolicyValueNet
from .ChessBoard import ChessBoard
from .AlphaZeroMCTS import AlphaZeroMCTS
from .SelfPlayDataset import SelfPlayDataset
import os

SelfPlayData = namedtuple(
    'SelfPlayData', ['piList', 'zList', 'FeaturePlanesList'])

class PolivyValueLoss(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,PHat, Value, Pi, zList):
        """_summary_

        Args:
            PHat (List): 对数动作概率向量 (N,boardlen**2)
            Value (int): 对每个局面的估值 (N,)
            Pi (_type_): `mcts`计算的得到的每个动作的概率向量   (N,boardlen**2)
            zList (_type_): 最终的游戏结果相对每一个玩家的奖励 (N,)

        Returns:
            _type_: _description_
        """
        value_loss = F.mse_loss(Value, zList) #均方误差函数 计算估值误差
        policy_loss = -torch.sum(Pi * PHat,dim=1).mean() #交叉熵函数 计算策略误差 
        return policy_loss + value_loss


class TrainModel:
    """训练模型"""
    
    def __init__ (self,boardlen=17,lr=0.1,SelfPlayCnt=10000,MCTSItersCnt=500,
                 FeaturePlanesCnt=4,batchsize=500,StartTrainSize=500,CheckFrequency=100,
                 TestGames=10,exploration=4,IsUsedGPU=False,IsSaveGame=False,*args, **kwargs) -> None:
        """_summary_

        Args:
            `boardlen` (int, optional): 棋盘边长长度. Defaults to 17.
            `lr` (float, optional): 学习率. Defaults to 0.1.
            `SelfPlayCnt` (int, optional): 自我博弈局数. Defaults to 10000.
            `MCTSItersCnt` (int, optional): 蒙特卡洛树搜索次数. Defaults to 500.
            `FeaturePlanesCnt` (int, optional): 特征平面个数. Defaults to 7.
            `batchsize` (int, optional): 批量大小. Defaults to 500.
            `StartTrainSize` (int, optional): 开始训练模型时的最小数据集大小. Defaults to 500.
            `CheckFrequency` (int, optional): 测试模型的频率. Defaults to 100.
            `TestGames` (int, optional): 测试模型时与历史最优模型的比赛局数. Defaults to 10.
            `exploration` (int, optional): 探索常数. Defaults to 4.
            `IsUsedGPU` (bool, optional): 是否使用GPU. Defaults to True.
            `IsSaveGame` (bool, optional): 是否保存子博弈的棋谱. Defaults to False.
        """
        self.boardlen = boardlen
        self.lr = lr
        self.SelfPlayCnt = SelfPlayCnt
        self.MCTSItersCnt = MCTSItersCnt
        self.FeaturePlanesCnt = FeaturePlanesCnt
        self.batchsize = batchsize
        self.StartTrainSize = StartTrainSize
        self.CheckFrequency = CheckFrequency
        self.TestGames = TestGames
        self.exploration = exploration
        self.IsUsedGPU = IsUsedGPU
        self.IsSaveGame = IsSaveGame
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and IsUsedGPU else "cpu")
        self.chessboard = ChessBoard(boardlen,FeaturePlanesCnt)

        self.net = self.__GetNet()
        self.mcts = AlphaZeroMCTS(self.net,self.exploration,self.MCTSItersCnt,self.exploration)
        
        #创建优化器和损失函数
        self.optimizer = optim.Adam(
            self.net.parameters(),lr=self.lr,weight_decay=1e-4)
        
        self.criterion = PolivyValueLoss()
        self.lr_scheduler =   MultiStepLR(                          #TODO
            self.optimizer,milestones=[1500,4000],gamma=0.1)      
        
        #创建数据集
        self.dataset = SelfPlayDataset(BoardLen=boardlen)
        
        # 记录数据
        self.trainLosses = self.__load_data('log/train_losses.json')
        self.game = self.__load_data('log/games.json')
    
    def __GetNet(self):
        """获取网络"""
        
        if not os.path.exists("model"):
            os.mkdir("model")
        if not os.path.exists("model/history"):
            os.mkdir("model/history")
        
        
        best_model = "model/best.pth"
        historyModel = sorted([i for i in os.listdir("model/history") if i.startswith("model")])
        
        print("🌍 开始加载网络!")
        if len(historyModel) > 0:
            best_model = "model/history/" + historyModel[-1]
            #从历史模型当中挑选换一个最好的模型
            net = torch.load(best_model)
            net.set_device(self.device)  #TODO
        else:
            net = PolicyValueNet(self.boardlen,self.FeaturePlanesCnt)
        print("🌈 网络加载完成!")
        
        return net
    
    def train(self):
        """训练模型"""
        for i in range(self.SelfPlayCnt):
            print(f"🌍 开始第{i+1}次自我博弈!")
            self.dataset.append(self.__SelfPlay())
            
            if len(self.dataset) >= self.StartTrainSize:
                dataLoader = DataLoader(self.dataset,batch_size=self.batchsize,shuffle=True,drop_last=False)
                print("🚽 开始训练模型!")
                
                self.net.train()
                feature_planes,pi,z = next(dataLoader)
                feature_planes = feature_planes.to(self.device)
                pi = pi.to(self.device)
                z = z.to(self.device)
                for epoch in range(5):
                    #前向传播
                    PHat,Value = self.net(feature_planes)
                    #反向传播
                    self.optimizer.zero_grad()
                    loss = self.criterion(PHat,Value,pi,z)  #计算损失
                    loss.backward() #反向传播
                    
                    self.optimizer.step()   #更新参数
                    self.lr_scheduler.step()    #更新学习率
                self.trainLosses.append(loss.item())    #记录损失

            if (i+1) % self.CheckFrequency == 0:
                self.__testModel()
            
    def __SelfPlay(self):
        """自我博弈一局
        
        Returns:
        """
        #初始化棋盘和数据容器
        self.net.eval()
        self.chessboard.ClearBoard()
        piList,FeaturePlanesList,players = [],[],[]
        actionList = []
        
        first = True
        #开始一局游戏
        while True:
            if first:
                action,pi = self.mcts.GetAction(self.chessboard)
                action = [action]
                first = False
            else:
                action1,pi = self.mcts.GetAction(self.chessboard)
                action2,pi = self.mcts.GetAction(self.chessboard)
                action = [action1,action2]
            FeaturePlanesList.append(self.chessboard.GetFeaturePlanes())
            piList.append(pi)
            players.append(self.chessboard.CurrentPlayer)
            actionList.append(action)
            self.chessboard.DoAction(action)
            #判断游戏是否结束:
            IsOver, winner = self.chessboard.IsGameOver()
            print("IsOver is ",IsOver)
            print("actions ", action)
            if IsOver:
                if winner is not None:
                    zList = [1 if i == winner else -1 for i in players]
                else:
                    zList = [0]*len(players)
                break
            
            #重置根节点:
        self.mcts.ResetRoot()
            
            #返回数据
        if self.IsSaveGame:
            self.game.append(actionList,players)
        
        selfplaydata = SelfPlayData(
            piList=piList,zList=zList,FeaturePlanesList=FeaturePlanesList)
        return selfplaydata
        
    
    
    def __testModel(self):
        os.makedirs("model",exist_ok=True)

        model = "model/best.pth"
        
        if not os.path.exists(model):
            torch.save(self.net,model)
            return 
        
        bestmodel = torch.load(model)
        bestmodel.eval()    #历史最好的模型
        bestmodel.set_device(self.device)
        mcts = AlphaZeroMCTS(bestmodel,self.exploration,self.MCTSItersCnt,self.exploration)
        
        self.mcts.SetSelfPlay(False)
        self.net.eval() #当前的模型
        
        print(" 开始比较模型性能!")
        
        WinCnt = 0
        for i in range(self.TestGames):
            self.chessboard.ClearBoard()
            self.mcts.ResetRoot()
            mcts.ResetRoot()
            first = True
            while True:
                IsOver,winner = self.__do_mcts_action(self.mcts)
                if IsOver:
                    Nwins += int(winner==ChessBoard.BLACK)
                    break
                
                isOver,winner = self.__do_mcts_action(mcts)
                if isOver:
                    break
            
            #如果胜率是50%以上，就保存当前模型
        if Nwins >= self.TestGames//2:
            torch.save(self.net,model)
            print("更新最新的模型")
        else:
            print("不更新最新的模型")
        
        self.mcts.IsSelfPlay(True)
    
    def save_model(self, model_name: str, loss_name: str, game_name: str):
        """ 保存模型

        Parameters
        ----------
        model_name: str
            模型文件名称，不包含后缀

        loss_name: str
            损失文件名称，不包含后缀

        game_name: str
            自对弈棋谱名称，不包含后缀
        """
        os.makedirs('model', exist_ok=True)

        path = f'model/{model_name}.pth'
        self.policy_value_net.eval()
        torch.save(self.policy_value_net, path)
        print(f'🎉 已将当前模型保存到 {os.path.join(os.getcwd(), path)}')

        # 保存数据
        with open(f'log/{loss_name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.train_losses, f)

        if self.is_save_game:
            with open(f'log/{game_name}.json', 'w', encoding='utf-8') as f:
                json.dump(self.games, f)
    
    
    def __do_mcts_action(self, mcts: AlphaZeroMCTS):
        """ 获取动作 """
        action = mcts.GetAction(self.chessboard)
        self.chessboard.DoAction(action)
        Isover, winner = self.chessboard.IsGameOver()
        return Isover, winner
    
    def __load_data(self, path: str):
        """ 载入历史损失数据 """
        data = []
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except:
            os.makedirs('log', exist_ok=True)

        return data
    