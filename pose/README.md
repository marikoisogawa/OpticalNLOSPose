# Human 3D Pose Estimation
Training and inference code for our CVPR2020 paper [[Optical Non-Line-of-Sight Physics-based 3D Human Pose Estimation](https://marikoisogawa.github.io/project/nlos_pose.html)].


## Environment

* **Supported OS:** MacOS, Linux.
* **License:**
  * You'll need a [Mujoco](http://www.mujoco.org/) licence to try our code.
* **Packages:**
  * Python >= 3.6
  * PyTorch >= 0.4 ([https://pytorch.org/](https://pytorch.org/))
  * gym ([https://github.com/openai/gym](https://github.com/openai/gym))
  * mujoco-py ([https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py))
  * OpenCV: ```conda install -c menpo opencv```
  * Tensorflow, OpenGL, yaml:
    ```conda install tensorflow pyopengl pyyaml```
* **Additional setup:**
  * For linux, the following environment variable needs to be set to greatly improve multi-threaded sampling performance:  
    ```export OMP_NUM_THREADS=1```
* **Note**: All scripts should be run from the root of this repo.


## Quick Demo

* Download our data and results including pretrained model from this [link](https://drive.google.com/file/d/1VA6NLOH8UpfzIxpXTxKiVw4C2jBOhsgM/view?usp=sharing). Place the unzipped datasets folder inside the folder as ```~/datasets/```, and place results folder as ```~/results/```.

* Try the following command to visualize the estimated results for in-the-wild Lindel et al's sequence:
```python ego_pose/ego_mimic_eval_wild.py --cfg demo_01 --iter 3000 --render --test-feat demo_01 --test-ind 30```

* Keyboard shortcuts for the visualizer: [keymap.md in Ye Yuan's repo](https://github.com/Khrylx/EgoPose/blob/master/docs/keymap.md)



## Training and Testing

* If you are interested in training and testing with our code, please see [train_and_test.md in Ye Yuan's repo](https://github.com/Khrylx/EgoPose/blob/master/docs/train_and_test.md) and try the same commands described in the repo. Our implementation uses this code and the training/testing processes are completely same as this repo.
* You can skip "forecasting" part. Try the state regression and ego-pose estimation parts only.

## Citation

If you find our work useful in your research, please consider citing our CVPR2020 paper! For further details, please check [our project page](https://marikoisogawa.github.io/project/nlos_pose.html).

```
@InProceedings{Isogawa_2020_CVPR,
author = {Isogawa, Mariko and Yuan, Ye and O'Toole, Matthew and Kitani, Kris M.},
title = {Optical Non-Line-of-Sight Physics-Based 3D Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020},
pages={7013-7022}
}
```

Also, our pose estimation training and inference code highly refers the following Yuan and Kitani's work, [Ego-Pose Estimation and Forecasting as Real-Time PD Control](https://www.ye-yuan.com/ego-pose):
```
@inproceedings{yuan2019ego,
  title={Ego-Pose Estimation and Forecasting as Real-Time PD Control},
  author={Yuan, Ye and Kitani, Kris},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  pages={10082--10092}
}
```