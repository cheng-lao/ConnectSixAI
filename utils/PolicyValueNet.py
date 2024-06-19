import torch 
from torch import nn
from torch.nn import functional as F
# from torch.nn import Module
from .ChessBoard  import ChessBoard

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self,in_channels:int,out_channels:int,Kernel :int ,Padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,Kernel,padding=Padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return F.relu(self.batchnorm(self.conv(x)))

class ResidueBlock(nn.Module):
    """残差块"""
    def __init__(self,in_channels=128,out_channels=128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                               kernel_size=3,padding=1,stride=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out + x)

class PolicyHead(nn.Module):
    def __init__(self,in_channels=128,board_len=17) -> None:
        super().__init__()
        self.board_len = board_len
        self.in_channels = in_channels
        self.conv = ConvBlock(in_channels,2,1)  # http://t.csdnimg.cn/TmEfm
        self.fc = nn.Linear(2*board_len**2, board_len**2)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))   # flatten in 1 dim
        return F.log_softmax(x,dim=1) # equivalent to log(softmax(x))
    
class ValueHead(nn.Module):
    """价值头"""

    def __init__(self,in_channels=128,board_len=9) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.board_len = board_len
        self.conv = ConvBlock(in_channels,1,1)
        self.fc = nn.Sequential(
            nn.Linear(board_len**2,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Tanh()   # tanh函数 双曲正切激活函数
        )
    
    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x
    
    
class PolicyValueNet(nn.Module):
    def __init__(self,board_len=17,FeaturePlanesCnt=6,UsedGPU=True) -> None:
        """网络初始化

        Args:
            `board_len` (int, optional): 棋盘大小(边长,默认是正方形). Defaults to 17.
            `FeaturePlanesCnt` (int, optional): 输入的通道数. Defaults to 6.
            `UsedGPU` (bool, optional): 是否使用GPU训练. Defaults to True.
        """
        super().__init__()
        self.board_len = board_len
        self.FeaturePlanesCnt = FeaturePlanesCnt
        self.UsedGPU = UsedGPU
        self.device = torch.device('cuda:0' if UsedGPU else 'cpu')
        self.conv = ConvBlock(FeaturePlanesCnt,128,3,Padding=1)
        self.residues = nn.Sequential(
            *[ResidueBlock(128,128) for _ in range(4)])
        self.PolicyHead = PolicyHead(128,board_len)
        self.ValueHead = ValueHead(128,board_len)
        
    def forward(self,x):
        """前向传播

        Args:
        -----
        `x`: Tesnor of shape(N, C, H, W)
        
        Returns:
        -------
        `PHat`: Tensor of shape (N, board_len^2)
            对数先验概率向量    决策头 输出一个一维向量，向量的长度是棋盘大小的平方，每个元素代表对应位置的落子概率

        `Value`: Tensor of shape (N, 1)
            当前局面的估值  价值头 输出一个一维向量，向量的长度是1，代表当前局面的估值
        """
        x = self.conv(x)
        x = self.residues(x)
        PHat = self.PolicyHead(x)
        Value = self.ValueHead(x)
        return PHat, Value
    
    def predict(self, ChessBoard: ChessBoard):  #TODO 再看看有没有细节问题
        """ 获取当前局面上所有可用 `action` 和他对应的先验概率 `P(s, a)`，以及局面的 `value`

        Parameters
        ----------
        chess_board: ChessBoard
            棋盘

        Returns
        -------
        probs: `np.ndarray` of shape `(len(chess_board.available_actions), )`
            当前局面上所有可用 `action` 对应的先验概率 `P(s, a)`

        value: float
            当前局面的估值
        """
        FeaturePlanes = ChessBoard.GetFeaturePlanes().to(self.device)
        FeaturePlanes.unsqueeze_(0)
        PHat, Value = self(FeaturePlanes)

        # 将对数概率转换为概率
        p = torch.exp(PHat).flatten()

        # 只取可行的落点
        if self.UsedGPU:
            p = p[ChessBoard.AvailableACtions].cpu().detach().numpy()
        else:
            p = p[ChessBoard.AvailableACtions].detach().numpy()

        return p, Value[0].item()

    def set_device(self, UsedGPU: bool):
        """ 设置神经网络运行设备 """
        self.UsedGPU = UsedGPU
        self.device = torch.device('cuda:0' if UsedGPU else 'cpu')