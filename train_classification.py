"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size in training') # 배치 사이즈 -> update하는 데이터의 사이즈!
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]') # 모델 설정 : 기본은 pointnet_cls
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40') # class 수
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training') # epoch 수
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training') # learning late -> Adam을 쓰니까 1e-3을 사용함을 알 수 있음.
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number') # input 포인트 개수?
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training') # optimizer 설정
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root') # 젤 중요한 것, 학습의 모든 결과 (log, 모델 가중치, 소스 코드)를 저장할 디렉토리 이름
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate') # Adam을 사용하니까 decay_late를 이용
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals') # 법선 벡터 정보를 이용하는지?
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline') # ??
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling') # ???
    return parser.parse_args()
## 학습 하이퍼파라미터 설정 


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval() # 모델을 평가 모드로 변경

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)): # 데이터로더로부터 배치 사이즈만큼의 points (포인트 클라우드 데이터)와 target (정답 레이블)을 받아와 루프를 돈다.

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1) # PointNet 모델은 (BatchSize, Features, NumPoints) 형태의 입력 vs DataLoader는 보통 (BatchSize, NumPoints, Features) : 그래서 차원을 바꿔줌
        pred, _ = classifier(points) # 모델에 points를 입력하여 예측을 수행
        pred_choice = pred.data.max(1)[1] # 로짓에서 제일 높은 값 추출해서 예측한 class 저장

        for cat in np.unique(target.cpu()): # 클래스별 정확도를 계산하는 부분
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum() # 전체 정확도를 계산하는 부분
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1] # 클래스별 정확도 최종 계산
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct) # 전체 정확도 최종 계산

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # ??

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) # 현재 시간을 기준으로 타임 스탬프 문자열을 만듦.
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    # log를 저장할 기본 경로
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr) # 지정해주지 않으면 타임스탬프 이름으로 된 이번 train 회차 전용 디렉토리를 만듦 (/log/classification)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)  # 지정됐으면 지정된 곳에 디렉토리로 고고
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')# 모델 가중치 (.pth)가 저장될 폴더
    checkpoints_dir.mkdir(exist_ok=True) 
    log_dir = exp_dir.joinpath('logs/') # 학습 로그(.txt)가 저장될 폴더
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args() # 하이퍼파라미터를 불러오고.. 
    logger = logging.getLogger("Model") # logging 모듈에서 "Model" 이라는 logger를 가져온다.
    logger.setLevel(logging.INFO) # "INFO" 등급의 메시지만 처리하도록 최소 등급을 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 로그 메시지의 형식을 정의
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model)) # 로그를 파일로 보내는 역할 -> log_dir 안에 "args.model".txt 파일을 생성
    file_handler.setLevel(logging.INFO) # 파일 핸들러도 INFO 등급만 설정
    file_handler.setFormatter(formatter) # 파일 핸들러가 위에서 만든 로그 형식(formatter)를 사용
    logger.addHandler(file_handler)  # "Model" 기록 계에 파일 핸들러를 부착 -> "Model" 기록계로 들어오는 INFO 등급 이상의 모든 메시지는 'pointnet_cls.txt' 파일에도 기록
    log_string('PARAMETER ...') # log_string 함수를 호출 -> logger에도 보내고 화면에도 출력
    log_string(args) # 모든 하이퍼파라미터 내용들을 화면과 logger에 보내서 파일에 기록함.

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/' # 학습 데이터의 파일 위치를 알려줌
    # from data_utils.ModelNetDataLoader에 있는 ModelNetDataLoader.py에서 힌트를 얻어옴!
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # 비율대로 나누는게 아니라 이미 제공되는 데이터셋이 train / test로 나누어져있음. -> 지정해놓은 거를 ModelNetDataLoader에서 처리를 해주고 넘겨주는 것
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # train/test DataLoader를 설정 -> tensor 구조로 만들어주는 것!

    '''MODEL LOADING'''
    num_class = args.num_category # 클래스 개수
    model = importlib.import_module(args.model) # args.model (예: pointnet_cls) 이름을 기반으로 models/pointnet_cls.py 파일을 동적으로 import
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir)) # 재현성을 위해 이번 학습에서 사용된 소스 코드(모델 파일, 유틸 파일, 이 train 파일)을 log 디렉토리에 복사해둠.

    classifier = model.get_model(num_class, normal_channel=args.use_normals) # 모델 뼈대 생성
    criterion = model.get_loss() # 모델에서 정의해놓은 Loss function을 가지고 옴.
    classifier.apply(inplace_relu) # ReLU 활성화함수를 메모리 최적화로 동작하게 변경

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda() # 모델과 Loss function을 GPU로 보냄.

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth', weights_only=False) # 파이토치 버전 업데이트 때문에 weight_only 인자 추가
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model') # 기존 설정 pre-trained 모델 불러오기

    except Exception as e:  # <-- "Exception as e" 추가
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!! 로드 중 심각한 오류 발생 !!!!!!!!!!!!!!")
        print(f"발생한 오류: {e}")  # <-- 진짜 오류 메시지 출력
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        ) # Optimizer 설정에 따라 Adam 옵티마이저를 초기화
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9) # 아니면 SGD 옵티마이저 사용

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7) # learning rate 스케줄러 사용 -> 20 epoch마다 lr을 0.7을 곱하여 학습률을 점차 줄임.
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0 # 초기화

    '''TRANING''' 

    logger.info('Start training...') ##### 제일 중요한 Training #####

    for epoch in range(start_epoch, args.epoch): # 설정한 Epoch까지 반복
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train() # model(classifier -> 역할이 분류기니까!)을 train() 모드로 설정

        scheduler.step() # 매 epoch 시작시 스케줄러를 업데이트 
        
        # mini-batch 단위로 데이터를 한 묶음 꺼냄, tqdm은 진행률을 보여줌 
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad() # 이전 mini-batch의 기울기(gradient) 값을 0으로 초기화

            # Data Augmentation 기법 
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3]) 
            points = torch.Tensor(points)
            points = points.transpose(2, 1) # (B,N,C) -> (B,C,N)으로 차원 변경

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points) # Forward path : 예측값(pred), PointNet의 T-Net 정규화를 위한 특수 출력값 (trans_feat)
            loss = criterion(pred, target.long(), trans_feat) # Loss 값을 계산
            pred_choice = pred.data.max(1)[1] # pred : 모델이 방금 처리한 mini-batch 데이터에 대한 로짓(logits) (배치 크기, 클래스 개수) -> 거기서 가장 클래스 중의 큰 점수를 찾아라!

            correct = pred_choice.eq(target.long().data).cpu().sum() # target : 이번 mini-batch의 실제 정답 label -> .eq()함수를 통해 하나씩 비교 [True, False, True, True, ...]처럼 맞았는지(True) 틀렸는지(False) 알려주는 리스트 -> CPU로 보낸 뒤 합치면 batch 중에 몇 개 맞혔는지 알 수 있음!
            mean_correct.append(correct.item() / float(points.size()[0])) # points tensor의 0번째 차원은 mini-batch size -> '맞힌 개수' / '배치 크기': 18 / 24.0 = 0.75. 즉, 이번 배치의 정확도를 계산
            loss.backward() # backward path : Loss에 대한 모델의 가중치 Gradient 계산
            optimizer.step() # 계산된 기울기 값을 이용해, 모델의 가중치를 실제로 업데이트 
            global_step += 1

        train_instance_acc = np.mean(mean_correct) # for 루프가 끝나면 (=모든 배치에 대해 학습이 끝나면) 정확도의 평균을 계산
        log_string('Train Instance Accuracy: %f' % train_instance_acc) 

        ## Vaildation 및 모델 저장
        with torch.no_grad(): # 검증 시에는 기울기 계산이 필요 없음.
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
            # test() 함수를 이용해 test dataset에 대한 평가 진행 -> 모델은 평가 모드로

            if (instance_acc >= best_instance_acc): # 이번 epoch의 정확도가 이전에 기록된 정확도보다 높다면.. 
                best_instance_acc = instance_acc # 갱신
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc): # 이번 epoch의 정확도가 이전에 기록된 정확도보다 높다면.. 
                logger.info('Save model...') 
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                } # 모델 가중치(model_state_dict), epoch 정보, 정확도, 옵티마이저 상태 등 학습의 모든 상태를 state 딕셔너리에 담음.
                torch.save(state, savepath) # 이 state 딕셔너리를 checkpoints/best_model.pth 파일로 저장
            global_epoch += 1

    logger.info('End of training...') # 모든 epoch가 지나면 종료


if __name__ == '__main__':
    args = parse_args() # 사용자가 터미널에서 입력한 학습 관련 설정값들을 불러온다. -> 터미널에서 --model (모델명) 과 같이 파라미터를 터미널에서 입력해줄 수 있다!
    main(args) # 설정값을 가지고 main을 호출하여 실제 학습을 실행
