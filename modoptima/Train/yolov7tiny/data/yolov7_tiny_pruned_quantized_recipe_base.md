<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---
# General Epoch/LR variables
num_epochs: &num_epochs {{num_epochs}}

# pruning hyperparameters
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch {{prun_start_epoch}}
pruning_end_epoch: &pruning_end_epoch {{prun_end_epoch}}
update_frequency: &pruning_update_frequency 0.5

prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.75
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.8
prune_high_target_sparsity: &prune_high_target_sparsity 0.85

# Quantization Epoch/LR variables
quantization_start_epoch: &quantization_start_epoch {{quant_start_ep}}
quantization_init_lr: &quantization_init_lr 0.0032 


# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs

pruning_modifiers:
  - !GMPruningModifier
    params:
      - model.7.conv.weight
      - model.9.conv.weight 
      - model.10.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_none_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - model.10.conv.weight
      - model.11.conv.weight
      - model.12.conv.weight
      - model.14.conv.weight
      - model.16.conv.weight
      - model.17.conv.weight
      - model.18.conv.weight
      - model.19.conv.weight
      - model.21.conv.weight
      - model.23.conv.weight
      - model.24.conv.weight
      - model.25.conv.weight
      - model.26.conv.weight
      - model.28.conv.weight
      - model.29.conv.weight
      - model.30.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - model.35.conv.weight
      - model.37.conv.weight
      - model.38.conv.weight
      - model.40.conv.weight
      - model.42.conv.weight
      - model.43.conv.weight
      - model.44.conv.weight
      - model.45.conv.weight
      - model.47.conv.weight
      - model.48.conv.weight


    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - model.50.conv.weight
      - model.52.conv.weight
      - model.53.conv.weight
      - model.54.conv.weight
      - model.55.conv.weight
      - model.57.conv.weight
      - model.58.conv.weight
      - model.60.conv.weight
      - model.61.conv.weight
      - model.62.conv.weight
      - model.63.conv.weight
      - model.65.conv.weight
      - model.66.conv.weight
      - model.68.conv.weight
      - model.69.conv.weight
      - model.70.conv.weight
      - model.73.conv.weight
      - model.75.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_high_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules:
      - model.0
      - model.1 
      - model.2
      - model.3 
      - model.4
      - model.5
      - model.7
      - model.9
      - model.10
      - model.11
      - model.12
      - model.14
      - model.15
      - model.16
      - model.17
      - model.18
      - model.19
      - model.21
      - model.23
      - model.24
      - model.25
      - model.26
      - model.28
      - model.29
      - model.30
      - model.35
      - model.37
      - model.38
      - model.40
      - model.42
      - model.43
      - model.44
      - model.45
      - model.47
      - model.48
      - model.50
      - model.52
      - model.53
      - model.54
      - model.55
      - model.57
      - model.58
      - model.60
      - model.61
      - model.62
      - model.63
      - model.65
      - model.66
      - model.68
      - model.69
      - model.70
      - model.71
      - model.73
      - model.74
      - model.75
      - model.76

---

# YOLOv7-tiny Pruned

This recipe creates a sparse, [YOLOv7].

When running, adjust hyperparameters based on training environment and dataset.


## Training


*script command:*

```
python train.py \
    --recipe ../recipes/yolov7_tiny_pruned_ver2.md \
    --weights PRETRAINED_WEIGHTS \
    --data voc.yaml \
    --hyp ../data/hyp.pruned.yaml \
```

hyp.pruned.yaml:
```yaml
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5 
cls_pw: 1.0 
obj: 1.0
obj_pw: 1.0 
iou_t: 0.20 
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5 
shear: 0.0  
perspective: 0.0 
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.05
copy_paste: 0.0
paste_in: 0.05
loss_ota: 1
```