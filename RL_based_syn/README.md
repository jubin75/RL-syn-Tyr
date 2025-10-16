# RL_based_syn
the framework based on reinforcement learning for forward synthesis

Code for the paper "Synthetically Feasible De Novo Molecular Design of Leads Based on a Reinforcement Learning Model: AI-Assisted Discovery of an Anti-IBD Lead Targeting CXCR4"

## Platform
This research is based on MolProphet: A One-Stop, General Purpose, and AI-Based Platform for the Early Stages of Drug Discovery

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/dw8h0BBQJvY/0.jpg)](https://www.youtube.com/watch?v=dw8h0BBQJvY)

[Website](https://www.molprophet.com) | [Video Introduction](https://www.youtube.com/watch?v=dw8h0BBQJvY) | [Paper](https://doi.org/10.1021/acs.jcim.3c01979)

## Installation
```
conda env create -f environment.yaml
conda activate rl_syn
```

## Datasets
You can download the processed data from this [link](https://drive.google.com/file/d/1Q5Pp_Ryj9DuUlVLNg0qGWDJg6dMXqqF5/view?usp=sharing)

## Model
Our model checkpoints can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1w2nswNiOW8wVezjcfiFbGbJQ0WGJQX00/view?usp=sharing)

## Prediction
Download and uncompress the model and processed data, then perform the following code
```
python demo.py
```

## Citation
Jiang, X., Lu, L., Li, J., Jiang, J., Zhang, J., Zhou, S., Wen, H., Cai, H., Luo, X., Li, Z., Wang, J., Ju, B., & Bai, R. (2024). Synthetically Feasible De Novo Molecular Design of Leads Based on a Reinforcement Learning Model: AI-Assisted Discovery of an Anti-IBD Lead Targeting CXCR4. In Journal of Medicinal Chemistry. American Chemical Society (ACS). https://doi.org/10.1021/acs.jmedchem.4c00184

## Reference
Yang, K., Xie, Z., Li, Z., Qian, X., Sun, N., He, T., Xu, Z., Jiang, J., Mei, Q., Wang, J., Qu, S., Xu, X., Chen, C., & Ju, B. (2024). MolProphet: A One-Stop, General Purpose, and AI-Based Platform for the Early Stages of Drug Discovery. In Journal of Chemical Information and Modeling (Vol. 64, Issue 8, pp. 2941–2947). American Chemical Society (ACS). https://doi.org/10.1021/acs.jcim.3c01979
