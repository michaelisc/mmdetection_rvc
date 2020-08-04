# MMDetection for the Robust Vision Challenge

This is an adaptation of MMDetection for the [ECCV2020](https://eccv2020.eu/) [Robust Vision Challenge](https://www.robustvision.net/).

### Features

- **RVC Dataset**

  Includes a RVC dataset using the annotations converted with the [RVC Toolkit](https://github.com/ozendelait/rvc_devkit).

- **Resample Datasets**

  Flexible dataset resampling [function](tools/resample_rvc_dataset.py)

- **Baseline Model**

  Includes [config files](configs/robust_vision_challenge/README.md) that were used to train the official object detection baseline submission.


## MMDetection README

The original MMDetection readme can be found in [MMDETECTION_README.md](MMDETECTION_README.md).


## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.


## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMDetection. There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).



## Citation

This toolbox is based on MMDetection, a flexible and powerful open source object detection toolkit. 
If you use is in your submission or research, please cite their project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE) building upon [MMDetection](https://github.com/open-mmlab/mmdetection).


## Contact

This repo is currently maintained by Claudio Michaelis ([@michaelisc](http://github.com/michaelisc)).
