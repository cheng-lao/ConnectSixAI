# coding: utf-8
from utils.train import TrainModel


# train_config = {
#     'lr': 1e-2,
#     'c_puct': 3,
#     'board_len': 9,
#     'batch_size': 500,
#     'is_use_gpu': False,
#     'n_test_games': 10,
#     'n_mcts_iters': 500,
#     'n_self_plays': 500, # 4000,
#     'is_save_game': False,
#     'n_feature_planes': 6,
#     'check_frequency': 100,
#     'start_train_size': 500
# }
train_model = TrainModel()
train_model.train()
