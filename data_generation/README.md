# Data Generation

## Preparation
* Please make your dataset folder as follows:
```
~/datasets/
```

* Download or clone this repository and add ```data_generation``` folder and it's sub folders to your Matlab path.

## Generate Synthetic Transient Image (Train/Test Data)
We synthesize transient images from depth images captured by Kinect (you can use any depth images).

#### Download source depth images
Please download [depth image zip](https://drive.google.com/file/d/1hwMcOH4KN_68z0kV3euE6M6dvTdltL8h/view?usp=sharing) and put the extracted folder as follows:
```
~/datasets/depth
```

#### Simple demo
Since the procedure for a whole sequence requires large memory size and computational time, I suggest to test the code once with the small set of frames (e.g., 200 frames). To start the demo with the small set of frames, set frame range like the followings (make sure to uncomment the one bellow line for non-demo use).
```
if strcmp(param.cfg_name, '0517_take_01') == true
    param.framerange_30fps = [1000, 1200];
    % param.framerange_30fps = [265+169-1,  265+2062+1]; % uncomment this for whole sequence
```
Then run the following command to generate transient images output under the ```~/datasets/transient/0517_take_01_poisson_tmblur_tmdown_sizenormdepth0512/``` folder.
```
data_augmentation_batch('0517_take_01', 512);
```
The former argument is a sequence name (you can use any name described in data_augmentation_batch.m), and the second argment defines how we shift the temporal peak of the transient images. We usually use five types of shift amount (i.e., 256, 384, 512, 640, 768).

### After the simple demo
Do the same things with the simple demo, with larger frame range. Set the range as follows,
```
if strcmp(param.cfg_name, '0517_take_01') == true
    % param.framerange_30fps = [1000, 1200];
    param.framerange_30fps = [265+169-1,  265+2062+1]; % uncomment this for whole sequence
```
And run the following. Then the transient images would be generated under the ```~/datasets/transient/0517_take_01_poisson_tmblur_tmdown_sizenormdepth0512/``` folder.
```
data_augmentation_batch('0517_take_01', 512);
```
Again, you can use any sequences described in data_augmentation_batch.m, and you might want to generate them five times with shifted temporal peak (i.e., 256, 384, 512, 640, 768). Please make sure that it requires huge memory size, computational time, and the generated transient images would have large file size if you run this for many sequences.


## Real Captured Transient Image (Test Data)

As a real captured data, we used transient measurements provided by [Lindell et al.](http://www.computationalimaging.org/publications/nlos-fk/). Please download their [interactive dataset](https://drive.google.com/open?id=1cb5augzU2Gh3M0CpQp3AKlN-C-N0HFI-).


## Mocap Data
The corresponded Mocap file (bvh) can be downloaded from [here](https://drive.google.com/file/d/1yyxd9cpRmnvYZuxrfmeQqjfi7ZTidM2Z/view?usp=sharing).

## Environment
We have tested our code with Matlab R2019a on macOSX 10.14.6.

## Citation

If you find our work useful in your research, please consider citing our CVPR2020 paper! For the paper and more information, please check [our project page](https://marikoisogawa.github.io/project/nlos_pose.html).

```
@InProceedings{Isogawa_2020_CVPR,
author = {Isogawa, Mariko and Yuan, Ye and O'Toole, Matthew and Kitani, Kris M.},
title = {Optical Non-Line-of-Sight Physics-Based 3D Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020},
pages={7013-7022}
}
```

If you use the interactive dataset (the real captured transient image), please cite this [Lindell et al.'s work](http://www.computationalimaging.org/publications/nlos-fk/):
```
@article{Lindell:2019:Wave,
author = {David B. Lindell and Gordon Wetzstein and Matthew O’Toole},
title = {Wave-based non-line-of-sight imaging using fast f-k migration},
journal = {ACM Trans. Graph. (SIGGRAPH)},
volume = {38},
number={4},
pages={116},
year = {2019},
}
```

Also, our data synthesis code highly refers the following [O'Toole et al.'s work](http://www.computationalimaging.org/publications/confocal-non-line-of-sight-imaging-based-on-the-light-cone-transform/):
```
@article{OToole:2018:ConfocalNLOS,
author = {Matthew O’Toole and David B. Lindell and Gordon Wetzstein},
title = {{Confocal Non-Line-of-Sight Imaging Based on the Light-Cone Transform}},
journal = {Nature},
year = {2018},
}
```
