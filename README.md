# GAutoMRI

Generalizable Reconstruction for Accelerating MR Imaging via Federated Learning with Neural Architecture Search
![image](/assets/GAutoMRI.png)

## Main Results

- In-Distribution
<img src="/assets/In-Distribution.png" alt="Description" width="600">

![image](/assets/In-Distribution.png)

- Out-of-Distribution
- <img src="/assets/Out-of-Distribution.png" alt="Description" width="600">

- Ablation study
<img src="/assets/Ablation.jpeg" alt="Description" width="500">

## Searching Phase

"""
python search.py --dataset fastmri --data-path /path/to/fastmri --output-dir /path/to/output --num-epochs 100 --num-classes 10 --num-samples 10 --num-architectures 10 --num-workers 4
"""

## Training Phase

"""
python train.py --dataset fastmri --data-path /path/to/fastmri --output-dir /path/to/output --num-epochs 100 --num-classes 10 --num-samples 10 --num-architectures 10 --num-workers 4
"""
