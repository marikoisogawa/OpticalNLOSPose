# Optical Non-Line-of-Sight Physics-based 3D Human Pose Estimation (CVPR2020)

![Loading image](teaser.png "teaser")

This repo contains the implementation of our paper:

Optical Non-Line-of-Sight Physics-based 3D Human Pose Estimation
Mariko Isogawa, Ye Yuan, Matthew O'Toole, Kris Kitani. CVPR 2020.

[[project page](https://marikoisogawa.github.io/project/nlos_pose.html)] [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Isogawa_Optical_Non-Line-of-Sight_Physics-Based_3D_Human_Pose_Estimation_CVPR_2020_paper.html)] [[video](https://www.youtube.com/watch?v=4HFulrdmLE8)]


## Generate Dataset
Please see [data generation code](https://github.com/marikoisogawa/OpticalNLOSPose/tree/master/data_generation) for our proposed transient image data synthesis and augmentation strategy based on depth data that can be transferred to a real-world NLOS imaging system.

## Pose Estimation
Please check [this code](https://github.com/marikoisogawa/OpticalNLOSPose/tree/master/pose) for our physics-based 3D pose estimation method from transient images. Our implementation highly refers [this repo for RL based pose estimation](https://github.com/Khrylx/EgoPose). Please also check the repo for the latest update.

## Citation
If you find our work useful in your research, please consider citing our paper:
```
@InProceedings{Isogawa_2020_CVPR,
author = {Isogawa, Mariko and Yuan, Ye and O'Toole, Matthew and Kitani, Kris M.},
title = {Optical Non-Line-of-Sight Physics-Based 3D Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020},
pages={7013--7022}
}
```

## License
The software in this repo is freely available for free non-commercial use. Please see the [license](https://github.com/marikoisogawa/OpticalNLOSPose/blob/master/LICENSE) for further details.