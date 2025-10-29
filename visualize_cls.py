import sys
import os
# 'models' 폴더의 경로를 파이썬이 찾을 수 있도록 추가합니다.
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

import torch
import numpy as np
import argparse
from models import pointnet_cls
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import matplotlib.pyplot as plt

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet Classification Visualization')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--data_path', default='data/modelnet40_normal_resampled/', help='path to data')
    
    # --- 시각화할 기본 개수를 10개로 변경 ---
    parser.add_argument('--num_vis', type=int, default=10, help='Number of samples to visualize')
    
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()

def main(args):
    # --- 1. 데이터셋 및 모델 로드 ---
    print('Load dataset ...')
    test_dataset = ModelNetDataLoader(root=args.data_path, args=args, split='test', process_data=False)
    
    print('Load model ...')
    model = pointnet_cls.get_model(k=args.num_category, normal_channel=args.use_normals)
    model = model.eval()
    
    model_path = f'log/classification/{args.log_dir}/checkpoints/best_model.pth'
    
    try:
        if torch.cuda.is_available():
            model.cuda()
            checkpoint = torch.load(model_path, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    # 클래스 이름 로드
    shape_names_file = f'{args.data_path}/modelnet{args.num_category}_shape_names.txt'
    shape_names = [line.rstrip() for line in open(shape_names_file)]

    # --- 2. 여러 개의 결과를 보여주기 위한 Plot 설정 ---
    num_to_show = args.num_vis
    # 격자 크기 계산 (예: 10개면 2x5)
    cols = 5
    rows = (num_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), subplot_kw={'projection': '3d'})
    axes = axes.flatten() # 2D 배열을 1D로 펴기

    print(f"\nVisualizing {num_to_show} random samples...")

    # --- 3. 여러 데이터에 대해 예측 및 시각화 반복 ---
    for i in range(num_to_show):
        ax = axes[i]
        
        # 무작위로 데이터 인덱스 선택
        item_index = np.random.randint(0, len(test_dataset))
        points, target = test_dataset[item_index]

        # 예측 수행
        with torch.no_grad():
            points_tensor = torch.from_numpy(points).unsqueeze(0).transpose(2, 1)
            if torch.cuda.is_available():
                points_tensor = points_tensor.cuda()
            
            pred, _ = model(points_tensor)
            predicted_choice = pred.data.max(1)[1].item()

        # 실제 정답과 예측 결과
        ground_truth = shape_names[target.item()]
        prediction = shape_names[predicted_choice]

        # 3D 산점도 그리기
        xyz_points = points[:, 0:3]
        ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2], s=15, c='blue', marker='.')
        
        # 제목 색상 설정 (예측이 틀리면 빨간색)
        title_color = 'green' if ground_truth == prediction else 'red'
        ax.set_title(f"Truth: {ground_truth}\nPrediction: {prediction}", color=title_color, fontsize=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # 남는 subplot 비활성화
    for i in range(num_to_show, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)