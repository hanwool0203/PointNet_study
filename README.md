# PointNet/PointNet++ Pytorch (개인 학습용 저장소)

이 저장소는 [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) 저장소를 기반으로 합니다.

원본 저장소의 코드를 개인적인 학습 목적으로 수정하고, 주석을 추가한 버전입니다. 훌륭한 코드를 공개해주신 원본 저자 [yanx27](https://github.com/yanx27)님께 감사드립니다.

## 이 저장소의 목적

PointNet 및 PointNet++ 모델의 Pytorch 구현을 깊이 있게 이해하기 위해 코드를 분석하고 주석을 다는 것을 목표로 합니다.

## 주요 변경 사항

* **상세 주석 추가**: 모델 구조, 데이터 처리, 학습 과정 등 코드의 주요 부분에 이해를 돕기 위한 상세한 주석을 추가했습니다.
* **코드 현대화**: (여기에 님이 수정한 내용을 구체적으로 적어주세요. 예: "최신 Pytorch 버전에 맞게 일부 deprecated 함수 수정", "Ubuntu 22.04 환경에서 실행되도록 의존성 파일 수정" 등)

## License

이 저장소는 원본 저장소와 동일하게 [MIT License](LICENSE)를 따릅니다.

---
---

## (이하 내용은 원본 저장소의 README입니다)

# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Update
**2021/03/27:** 

(1) Release pre-trained models for semantic segmentation, where PointNet++ can achieve **53.5\%** mIoU.

(2) Release pre-trained models for classification and part segmentation in `log/`.

**2021/03/20:** Update codes for classification, including:

(1) Add codes for training **ModelNet10** dataset. Using setting of ``--num_category 10``. 

(2) Add codes for running on CPU only. Using setting of ``--use_cpu``. 

(3) Add codes for offline data preprocessing to accelerate training. Using setting of ``--process_data``. 

(4) Add codes for training with uniform sampling. Using setting of ``--use_uniform_sample``. 

**2019/11/26:**

(1) Fixed some errors in previous codes and added data augmentation tricks. Now classification by only 1024 points can achieve **92.8\%**! 

(2) Added testing codes, including classification and segmentation, and semantic segmentation with visualization. 

(3) Organized all models into `./models` files for easy using.

## Install
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification (ModelNet10/40)
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
You can run different modes with following codes. 
* If you want to use offline processing of data, you can use `--process_data` in the first run. You can download pre-processd data [here](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing) and save it in `data/modelnet40_normal_resampled/`.
* If you want to train on ModelNet10, you can use `--num_category 10`.
```shell
# ModelNet40
## Select different models in ./models 

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg

## e.g., pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal

## e.g., pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps

# ModelNet10
## Similar setting like ModelNet40, just using --num_category 10

## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 10
python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
```

### Performance
| Model | Accuracy |
|--|--|
| PointNet (Official) |  89.2|
| PointNet2 (Official) | 91.9 |
| PointNet (Pytorch without normal) |  90.6|
| PointNet (Pytorch with normal) |  91.4|
| PointNet2_SSG (Pytorch without normal) |  92.2|
| PointNet2_SSG (Pytorch with normal) |  92.4|
| PointNet2_MSG (Pytorch with normal) |  **92.8**|

## Part Segmentation (ShapeNet)
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_msg
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```
### Performance
| Model | Inctance avg IoU| Class avg IoU 
|--|--|--|
|PointNet (Official)	|83.7|80.4	
|PointNet2 (Official)|85.1	|81.9	
|PointNet (Pytorch)|	84.3	|81.1|	
|PointNet2_SSG (Pytorch)|	84.9|	81.8	
|PointNet2_MSG (Pytorch)|	**85.4**|	**82.5**	


## Semantic Segmentation (S3DIS)
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/stanford_indoor3d/`.
### Run
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```
Visualization results will save in `log/sem_seg/pointnet2_sem_seg/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).

### Performance
|Model  | Overall Acc |Class avg IoU | Checkpoint 
|--|--|--|--|
| PointNet (Pytorch) | 78.9 | 43.7| [40.7MB](log/sem_seg/pointnet_sem_seg) |
| PointNet2_ssg (Pytorch) | **83.0** | **53.5**| [11.2MB](log/sem_seg/pointnet2_sem_seg) |

## Visualization
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)


## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```
```
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      journal={ECCV}
}
```
## Selected Projects using This Codebase
* [PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
* [On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
* [Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
* [PCT: Point Cloud Transformer](https://github.com/MenghaoGuo/PCT)
* [PSNet: Fast Data Structuring for Hierarchical Deep Learning on Point Cloud](https://github.com/lly007/PointStructuringNet)
* [Stratified Transformer for 3D Point Cloud Segmentation, CVPR'22](https://github.com/dvlab-research/stratified-transformer)
