import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

# 완성차 : 분류 모델이라고 봐도 됨.
# 결국에는 논문 보면 Max pooling하고 (B, 1024)이후 구간임!

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel: # 법선 벡터 쓰면 input 채널이 6으로 됨.
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel) # PointNetEncoder로 핵심 Backbone을 돌림!
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k) # 전역 특징을 받아 분류를 수행하는 3단 MLP (FC layer)

        self.dropout = nn.Dropout(p=0.4) # 과적합 방지
        self.bn1 = nn.BatchNorm1d(512) # Gradient Vanishing 문제를 해결하기 위한 BN
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x) # 유틸 함수의 Forward path를 쫙 돌림.
        x = F.relu(self.bn1(self.fc1(x))) # (B, 1024) -> (B, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # (B, 512) -> (B, 256) (드롭아웃 적용)
        x = self.fc3(x) # (B, 256) -> (B, k) (최종 클래스 점수(Logits) 출력)
        x = F.log_softmax(x, dim=1) # 최종 로짓을 softmax를 통해 0~1의 확률값으로 -> get_loss에서 nll_loss를 적용하기 위함.
        return x, trans_feat # 최종 예측 x (B, k)와, 손실 계산에 필요한 trans_feat (B, 64, 64)를 반환 (정규화 Loss)

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale # T-Net 정규화 손실의 '중요도'를 조절하는 하이퍼파라미터

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target) # 기본 Loss : 모델의 pred - 실제 정답 target과 얼마나 다른지 계산
        mat_diff_loss = feature_transform_reguliarzer(trans_feat) # T-Net 정규화 Loss를 계산 (Regularization)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale # '총 손실' = (기본 분류 손실) + (T-Net 페널티 * 중요도)
        return total_loss # 이 total_loss를 반환하여 train 함수에서 backpropagtion을 하고 파라미터를 업데이트 하는 것!
