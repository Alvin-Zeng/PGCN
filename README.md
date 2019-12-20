## Graph Convolutional Networks for Temporal Action Localization



This repo holds the codes and models for the PGCN framework presented on ICCV 2019

**Graph Convolutional Networks for Temporal Action Localization**
Runhao Zeng*, Wenbing Huang*, Mingkui Tan, Yu Rong, Peilin Zhao, Junzhou Huang, Chuang Gan,  *ICCV 2019*, Seoul, Korea.

[[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zeng_Graph_Convolutional_Networks_for_Temporal_Action_Localization_ICCV_2019_paper.pdf)


## Updates


20/12/2019 We have uploaded the RGB features, trained models and evaluation results! We found that increasing the number of proposals to 800 in the testing
further boosts the performance on THUMOS14. We have also updated the proposal list.


# Contents
----

* [Usage Guide](#usage-guide)
   * [Prerequisites](#prerequisites)
   * [Code and Data Preparation](#code-and-data-preparation)
      * [Get the code](#get-the-code)
      * [Download Datasets](#download-datasets)
      * [Download Features](#download-features)
   * [Training PGCN](#training-pgcn)
   * [Testing PGCN](#testing-trained-models)
* [Other Info](#other-info)
   * [Citation](#citation)
   * [Contact](#contact)


----
# Usage Guide

## Prerequisites
[[back to top](#graph-convolutional-networks-for-temporal-action-localization)]

The training and testing in PGCN is reimplemented in PyTorch for the ease of use. 

- [PyTorch 1.0.1][pytorch]
                   
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

 
 
## Code and Data Preparation
[[back to top](#graph-convolutional-networks-for-temporal-action-localization)]

### Get the code

Clone this repo with git, **please remember to use --recursive**

```bash
git clone --recursive https://github.com/Alvin-Zeng/PGCN
```

### Download Datasets

We support experimenting with two publicly available datasets for 
temporal action detection: THUMOS14 & ActivityNet v1.3. Here are some steps to download these two datasets.

- THUMOS14: We need the validation videos for training and testing videos for testing. 
You can download them from the [THUMOS14 challenge website][thumos14].
- ActivityNet v1.3: this dataset is provided in the form of YouTube URL list. 
You can use the [official ActivityNet downloader][anet_down] to download videos from the YouTube. 


### Download Features

Here, we provide the I3D features (RGB+Flow) for training and testing. You can download it from [Google Cloud][features_google] or [Baidu Cloud][features_baidu].


## Training PGCN
[[back to top](#graph-convolutional-networks-for-temporal-action-localization)]

Plesse first set the path of features in data/dataset_cfg.yaml

```bash
train_ft_path: $PATH_OF_TRAINING_FEATURES
test_ft_path: $PATH_OF_TESTING_FEATURES
```


Then, you can use the following commands to train PGCN

```bash
python pgcn_train.py thumos14 --snapshot_pre $PATH_TO_SAVE_MODEL
```

After training, there will be a checkpoint file whose name contains the information about dataset and the number of epoch.
This checkpoint file contains the trained model weights and can be used for testing.

## Testing Trained Models
[[back to top](#graph-convolutional-networks-for-temporal-action-localization)]



You can obtain the detection scores by running 

```bash
sh test.sh TRAINING_CHECKPOINT
```

Here, `TRAINING_CHECKPOINT` denotes for the trained model.
This script will report the detection performance in terms of [mean average precision][map] at different IoU thresholds.

The trained models and evaluation results are put in the "results" folder.

You can obtain the two-stream results on THUMOS14 by running
```bash
sh test_two_stream.sh
```

### THUMOS14

| mAP@0.5IoU (%)                    | RGB   | Flow  | RGB+Flow      |
|-----------------------------------|-------|-------|---------------|
| P-GCN (I3D)                       | 37.23 | 47.42 | 49.07 (49.64) |


#####Here, 49.64% is obtained by setting the combination weights to Flow:RGB=1.2:1 and nms threshold to 0.32



# Other Info
[[back to top](#graph-convolutional-networks-for-temporal-action-localization)]

## Citation


Please cite the following paper if you feel PGCN useful to your research

```
@inproceedings{PGCN2019ICCV,
  author    = {Runhao Zeng and
               Wenbing Huang and
               Mingkui Tan and
               Yu Rong and
               Peilin Zhao and
               Junzhou Huang and
               Chuang Gan},
  title     = {Graph Convolutional Networks for Temporal Action Localization},
  booktitle   = {ICCV},
  year      = {2019},
}
```

## Contact
For any question, please file an issue or contact
```
Runhao Zeng: runhaozeng.cs@gmail.com
```



[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads
[tdd]:https://github.com/wanglimin/TDD
[anet]:https://github.com/yjxiong/anet2016-cuhk
[faq]:https://github.com/yjxiong/temporal-segment-networks/wiki/Frequently-Asked-Questions
[bs_line]:https://github.com/yjxiong/temporal-segment-networks/blob/master/models/ucf101/tsn_bn_inception_flow_train_val.prototxt#L8
[bug]:https://github.com/yjxiong/caffe/commit/c0d200ba0ed004edcfd387163395be7ea309dbc3
[tsn_site]:http://yjxiong.me/others/tsn/
[custom guide]:https://github.com/yjxiong/temporal-segment-networks/wiki/Working-on-custom-datasets.
[thumos14]:http://crcv.ucf.edu/THUMOS14/download.html
[tsn]:https://github.com/yjxiong/temporal-segment-networks
[anet_down]:https://github.com/activitynet/ActivityNet/tree/master/Crawler
[map]:http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
[action_kinetics]:http://yjxiong.me/others/kinetics_action/
[pytorch]:https://github.com/pytorch/pytorch
[ssn]:http://yjxiong.me/others/ssn/
[untrimmednets]:https://github.com/wanglimin/UntrimmedNet
[emv]:https://github.com/zbwglory/MV-release
[features_google]: https://drive.google.com/open?id=1C6829qlU_vfuiPdJSqHz3qSqqc0SDCr_
[features_baidu]: https://pan.baidu.com/s/1Dqbcm5PKbK-8n0ZT9KzxGA
