_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/rvc_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model
model = dict(roi_head=dict(bbox_head=dict(num_classes=640, reg_class_agnostic=True)))

# data
data_root = 'data/rvc/'
data = dict(train=dict(ann_file=data_root + 'joined_train_boxable_sampled.json'),
            val=dict(ann_file=data_root + 'joined_val_boxable_tiny.json'),
            test=dict(ann_file=data_root + 'joined_val_boxable_tiny.json'))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2])
total_epochs = 3
