# GAutoMRI

Generalizable Reconstruction for Accelerating MR Imaging via Federated Learning with Neural Architecture Search
![image](/assets/GAutoMRI.png)

## Main Results

- In-Distribution

![image](/assets/In-Distribution.png)

- Out-of-Distribution
![image](/assets/Out-of-Distribution.png)

- Ablation study
<img src="/assets/Ablation.jpeg" alt="Description" width="300" height="200">

![image](/assets/Ablation.jpeg)

## Searching Phase

"""
python search.py --dataset fastmri --data-path /path/to/fastmri --output-dir /path/to/output --num-epochs 100 --num-classes 10 --num-samples 10 --num-architectures 10 --num-workers 4
"""

## Training Phase

"""
python train.py --dataset fastmri --data-path /path/to/fastmri --output-dir /path/to/output --num-epochs 100 --num-classes 10 --num-samples 10 --num-architectures 10 --num-workers 4
"""
