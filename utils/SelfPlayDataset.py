import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque , namedtuple
from torch import Tensor

# class SelfPlayData():
#     pass

SelfPlayData = namedtuple(
    'SelfPlayData', ['piList', 'zList', 'FeaturePlanesList'])

class SelfPlayDataset(Dataset):
    def __init__(self,BoardLen) -> None:
        super().__init__()
        self.__dataQueue = deque(maxlen=10000)
        self.boardlen = BoardLen
        # self.BoardLen = BoardLen    # 棋盘的边长(默认棋盘是长宽相等)
        
        
    def __len__(self):
        return len(self.__dataQueue)
    
    def __getitem__(self, index) :
        return self.__dataQueue[index]
    
    def append(self,SelfPlayData:SelfPlayData):
        """像数据集当中插入数据

        Args:
            `self_play_data.z_list`: 储存一局之中每个动作的玩家相对最后的游戏结果的奖赏列表
            `self_play_data.feature_planes`: 一局之中每个动作对应的特征平面组成的列表
            `self_play_data.pi`:  棋盘当中每个动作的概率值
        """
        n = self.boardlen
        zlist = torch.Tensor(SelfPlayData.zList)
        pilist = SelfPlayData.piList
        featurePlanesList = SelfPlayData.FeaturePlanesList
        # 通过翻转和镜像扩充已有的数据集
        for z, pi, FeaturePlanes in zip(zlist,pilist,featurePlanesList):
            for i in range(4):
                # 逆时针旋转90°
                rotFeatures = torch.rot90(Tensor(FeaturePlanes),i,(1,2))
                rotPi = torch.rot90(Tensor(pi.reshape(n,n),i))
                self.__dataQueue.append(
                    (rotFeatures,rotPi.flatten(),z)
                )
                
                # 对逆时针旋转后的数组进行水平反转
                flipFeatures = torch.flip(rotFeatures,[2])
                flipPi = torch.fliplr(rotPi)
                self.__dataQueue.append(
                    (flipFeatures,flipPi.flatten(),zlist)
                )
    
    
    
    