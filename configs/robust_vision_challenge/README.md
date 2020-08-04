# Robust Vision Challenge

## Introduction
Baseline model for the [ECCV2020](https://eccv2020.eu/) [Robust Vision Challenge](https://www.robustvision.net/).

## model

| Model          | Backbone |  Style  | Normalization | Lr schd | x COCO | x OID | x MVD | Config |
| :------------: | :------: | :-----: | :-----------: | :-----: | :----: | :---: | :---: | :----: |
|FRCNN_R50_GN_RVC| R-50-FPN | pytorch | GN+WS         |  3e     | 2      | 0.4   | 10    | [config](faster_rcnn_r50_fpn_gn_ws-all_3e_rvc-sampled.py) | 

x COCO/OID/MVD indicates the corresponding resampling factors

