import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# 사실상 엔진룸이라고 보면 된다!

# T-Net : 입력된 포인트 클라우드를 표준적인 자세로 알아서 회전/정렬시켜주는 작은 네트워크
# 목적 : 입력된 (B,3,N) 크기의 포인트 클라우드를 정렬하기 위한 [3x3] 크기의 변환 행렬을 예측
# 구조 : 사실상 미니 PointNet

class STN3d(nn.Module): # Input T-Net
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) # 1D Convolution으로 차원을 (3->64->128->1024로 늘림)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # Shared-MLP 부분 : for를 돌면서 각 point(N개)에 대해 Linear()를 적용하는게 아니라 Conv1d의 필터 개수(64개, 128개, 1024개)로 N개의 모든 점에 대해 동일한 가중치로 연산을 병렬로 처리
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # FC layer를 거치면서 최종적으로 (B,9)로 만듬.
        self.relu = nn.ReLU()


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0] # tensor 2번 차원(N)에 대해 최댓값을 찾는다. -> keepdim=True (차원을 유지하고 크기를 1로 만듬)
        # Shared-MLP 부분을 지나면 대칭함수 : Max Pooling을 통해 전역 feature를 만듬.
        x = x.view(-1, 1024) # nn.Linear에 넣기 위해 2D tensor로 변경

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # 네트워크가 예측한 9개 숫자에 항등 행렬을 더해줌 -> 학습 초기에는 네트워크가 항등행렬에서 출발해야 수렴이 안정적이고 빠름
        x = x.view(-1, 3, 3) # 최종 출력 : (B, 3, 3)의 변환 행렬
        return x

# Feature T-Net : 위의 STN3d와 비슷함. 차원수만 다른 것!

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

# Backbone : feature 추출기
# global_feat: True이면 분류용 (B, 1024) 전역 특징을 반환하고, False이면 세그멘테이션용 (B, 1088, N) 개별 점 특징을 반환
# feature_transform: True이면 STNkd (Feature T-Net)를 사용

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3): 
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel) # 3D Input T-Net을 장착
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) # Shared-MLP 부분
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64) # 64차원 Feature T-Net을 장착

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x) # (B, 3, 3) Input T-Net 행렬을 얻음 (변환 행렬)
        x = x.transpose(2, 1) # (B, N, 3) 형태로 바꿔줌
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans) # torch.bmm을 사용하여 x와 변환행렬(trans)을 곱함 -> input 포인트 클라우드가 정렬
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1) # 다시 (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x))) # 첫 번째 Shared-MLP을 거쳐 (B,64,N) 크기의 특징을 뽑는다.

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat) # feature transform 행렬을 곱해주어 feature 공간을 정렬
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x # (B, 64, N)의 개별 포인트의 feature를 저장 -> 나중에 Segmentation에서 이용!
        x = F.relu(self.bn2(self.conv2(x))) # (B, 64, N) -> (B, 128, N)
        x = self.bn3(self.conv3(x)) # (B, 128, N) -> (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0] # Max Pooling을 통해 N개의 point 중에서 제일 큰 값만 뽑는다.
        x = x.view(-1, 1024) # (B, 1024)로 reshape
        if self.global_feat:
            return x, trans, trans_feat # classification용을 보냄
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat # Segmentation용 (B, 1024, N)으로 복제한 뒤, 아까 저장해둔 pointfeat (B, 64, N)와 합쳐 (B, 1088, N) 크기의 텐서를 반환

# 정규화 Loss 
# 목적 : STNkd가 예측한 (B, 64, 64) 행렬 trans가 '좋은' 변환, 즉 **직교 행렬(Orthogonal Matrix)**에 가깝도록 강제하는 손실 함수
# 이유 : 찌그러트릴 수 있는 Scaling, Shearing을 막기 위해서!

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :] # [64x64] 크기의 항등행렬을 만듬.
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))) # A * A^T - I를 계산 -> 완벽한 직교 행렬이면 이 결과는 0에 가까운 행렬 -> 이 '차이 행렬'의 크기(L2 Norm)를 계산 -> batch 전체의 평균 손실을 구한다. 
    return loss
