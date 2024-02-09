# Towards Efficient and Exact Optimization of Language Model Alignment

## Overview

This is the official pytorch implementation of the EXO algorithm for *efficient exact optimization* of aligning language models (LMs) with human preferences, as described in [Towards Efficient and Exact Optimization of Language Model Alignment](https://arxiv.org/pdf/2402.00856.pdf). 

<div align="center">
  <img src="imgs/rkl_fkl.png" width="400px" />
</div>

EXO essentially minimizes the reverse KL between the empirical distributions defined by the policy and the reward. As a comparison, DPO corresponds to minimizing the forward KL. The following figure illustrates the distinct behavior of policies obtained by minimizing (a) the reverse KL and (b) the forward KL.

The codes will be coming soon!

## Citing

```
@inproceedings{Ji2024TowardsExact,
  title={Towards Efficient and Exact Optimization of Language Model Alignment},
  author={Haozhe Ji, Cheng Lu, Yilin Niu, Pei Ke, Hongning Wang, Jun Zhu, Jie Tang, Minlie Huang},
  year={2024},
  url={https://arxiv.org/abs/2402.00856}
}
```

Please kindly cite our work if you find the paper or this repository useful :)

