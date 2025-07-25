# MvP

## Introduction

We provide the config files for MvP: [Direct multi-view multi-person 3d pose estimation](https://arxiv.org/pdf/2111.04076.pdf).

[Official Implementation](https://github.com/sail-sg/mvp)

```BibTeX
@article{zhang2021direct,
  title={Direct multi-view multi-person 3d pose estimation},
  author={Zhang, Jianfeng and Cai, Yujun and Yan, Shuicheng and Feng, Jiashi and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={13153--13164},
  year={2021}
}
```

## Results and Models

We evaluate MvP on 3 popular benchmarks, report the Percentage of Correct Parts (PCP), Mean Per Joint Position Error (MPJPE), MPJPE with Procrustes Analysis (PA) as PA-MPJPE, Probability of Correct Keypoint (PCK), mAP and recall on Campus, Shelf and CMU Panoptic dataset.

To be more fair in evaluation, some modifications are made compared to the evaluations in the original work. For PCP, instead of by body parts, we evaluate by the limbs defined in `selected_limbs_names` and `additional_limbs_names`. We remove the root alignment in MPJPE and provide PA-MPJPE instead. Thresholds for outliers are removed as well.


### Campus

MvP for Campus fine-tuned from the model weights pre-trained with 3 selected views in CMU Panoptic dataset is provided. Fine-tuning with the model pre-train with CMU Panoptic HD camera view 3, 6, 12 gives the best final performance on Campus dataset.

| Config | PCP |  MPJPE(mm) |PA-MPJPE(mm)| PCK@50 | PCK@100 |Download |
|:------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [mvp_campus.py](./campus_config/mvp_campus.py) | 96.46 | 64.09 | 50.07 | 58.91| 94.14 |[model](https://drive.google.com/file/d/1CqZcl77qRR1gf3cm33IPF0R4HDSTsn4T/view?usp=sharing) |


### Shelf

MvP for Shelf fine-tuned from the model weights pre-trained with 5 selected views in CMU Panoptic dataset is provided. The 5 selected views, HD camera view 3, 6, 12, 13 and 23 are the same views used in VoxelPose.

| Config | PCP |  MPJPE(mm) |PA-MPJPE(mm)| PCK@50 | PCK@100 |Download |
|:------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [mvp_shelf.py](./shelf_config/mvp_shelf.py)  | 96.77 | 52.18 | 40.95 | 74.17 | 97.80 | [model](https://drive.google.com/file/d/1gnrOVwvjvtisr9kYJfCFaoDd_XK6Pdsw/view?usp=sharing)  |


### CMU Panoptic

MvP for CMU Panoptic trained from stcratch with pre-trained Pose ResNet50 backbone is provided. The provided model weights were trained and evaluated with the 5 selected views same as VoxelPose (HD camera view 3, 6, 12, 13, 23).  A checkpoint trained with 3 selected views (HD camera view 3, 12, 23) is also provided as the pre-trained model weights for Campus dataset fine-tuning.

| Config | AP25 | AP100 | Recall@500 | PCP | MPJPE(mm) | PA-MPJPE(mm)|PCK@50 | PCK@100 |Download |
|:------:|:----:|:----:|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [mvp_panoptic.py](./panoptic_config/mvp_panoptic.py) | 90.67 | 96.93 | 99.52 | 99.28 | 18.64 | 17.07 | 97.37| 99.08| [model](https://drive.google.com/file/d/1Z4JhEKSxd9MMimPWR6nmwsrzfNgqYAhx/view?usp=sharing) |
| [mvp_panoptic_3cam.py](./panoptic_config/mvp_panoptic_3cam.py) | 56.12 | 94.71 | 98.59 | 97.13 |39.25 | 29.65 | 88.85 | 95.10| [model](https://drive.google.com/file/d/1JIrY1y_Lyd55KJ6PIhE7Kvz7_I6JqI2k/view?usp=sharing)  |

### Pose ResNet50 Backbone

All the checkpoints provided above were trained on top of the pre-trained [Pose ResNet50](https://drive.google.com/file/d/1Ars5rH0Ryz1CqbfRItyPB-aDomS5gaGo/view?usp=sharing) backbone weights.
