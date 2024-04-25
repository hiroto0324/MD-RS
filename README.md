# MD-RS (Under Review)

## Architecture
<img src="figures/fig_schematic_MD-RS.svg" width="800">

## Overview
![demo](figures/fig_demonstration_MD-RS.svg)

## Mixed Time Constants

In standard RC models, reservoir neurons typically share a common time constant $\tau$. However, in this study, we introduce neurons with diverse time constants into the reservoir, in order to enhance the flexibility of the reservoir's response to inputs [Perez+, 2021] [Tanaka+, 2022].
The implementation is detailed as follows:
$$
\Gamma \dot{\mathbf{x}} = - \mathbf{x} + J \mathbf{\phi}(\mathbf{x}) + V \mathbf{f}^{\rm in},
$$
where $\Gamma = \text{diag}(\tau_1, \tau_2, \ldots, \tau_N)$.

In this study, the time constants $\tau_1, \tau_2, \ldots, \tau_N$ are determined as follows:
1.  A small time constant $\tau_\text{S}$ and a large time constant $\tau_\text{L}$ with $\tau_\text{S}<\tau_\text{L}$ are established.
2. The $\tau$-mixing ratio $r\in[0, 1]$ is set.
3. The ratio of the number of neurons with time constant $\tau_\text{S}$ to those with $\tau_\text{L}$ is adjusted to be $(1-r):r$.
4. The assignment of $\tau_\text{S}$ or $\tau_\text{L}$ to $\tau_1, \tau_2, \ldots, \tau_N$ is randomized.

## Common $\tau$ vs. Mixed $\tau$
For demonstration, we compare two common $\tau$ models: the Small $\tau$ model ($\tau = 0.01$) and the Large $\tau$ model ($\tau = 0.025$), along with a Mixed $\tau$ model ($\tau_{\rm S}=0.01, \tau_{\rm L}=0.025, r=0.9$). 

According to the magnitude of $\tau$, the reservoir exponentially forgets its past states. 

The Small $\tau$ reservoir cannot retain past inputs for long, making it unsuitable for capturing long-term dependencies in anomalies. However, it quickly returns to a normal state once the anomalous input ends. 

Conversely, the Large $\tau$  reservoir can retain past inputs over extended periods, which is advantageous for detecting anomalies with long-term dependencies. Nevertheless, it has the drawback of taking a long time to return to a normal state after the end of an anomalous input.

The Mixed $\tau$ reservoir is a hybrid that incorporates the best features of both the Small $\tau$ and Large $\tau$ reservoirs.
| Model |Capturing long-term dependency| Rapid return to normal state |
| ---- | ---- | ---- |
|Small $\tau$| $\times$  | $\checkmark$ | 
|Large $\tau$| $\checkmark$ | $\times$ |
|Mixed $\tau$ (ours)| $\checkmark$ | $\checkmark$|

<div style="text-align: center;"> 
<img src="figures/fig_demo_different-tau_UCR_139.svg" width="500">
<img src="figures/fig_demo_different-tau_UCR_133.svg" width="500">
</div>

# Get Started

1. Install Python ..., PyTorch >= ....
2. Download data. You can obtain all benchmarks from [Google Cloud](). All the datasets are well pre-processed.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder  `./scripts`. You can reproduce the experiment results as follows:


# Main Results

## Note: Illusion of Performance in Point Adjustment (PA)
In [the code for DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/tree/main), the original anomaly scores are first converted into binary prediction labels using a thresholding operation.
After this, the true anomaly labels are referred to in order to perform Point Adjustment (PA) on the sequence (referred to as the PA sequence), which is then used as the new anomaly score.
Performance metrics such as Affiliation precision/recall, AUC, Range AUC, and VUS are calculated based on this PA sequence in [the code for DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/tree/main).

However, as shown in the example below, the PA sequence almost becomes identical to the true anomaly label sequence, resulting in performance metrics that are overly optimistic.

![fig_demo_UCR_139_DCdetector](figures/fig_demo_UCR_139_DCdetector.svg)

| Method                      | AUC ROC     | PR AUC      | Max F1 Score |
|-----------------------------|-------------|-------------|--------------|
| DCdetector (Anomaly Score)  | 0.499013335 | 0.037474823 | 0.025104603  |
| DCdetector (PA sequence)    | 0.998706897 | 0.965116279 | 0.963855422  |


## Overall Results

All of these results were calculated **without point adjustment (PA)**.
PA tends to make results appear more optimistic than they should, as shown above.

### UCR
![UCR Results](figures/UCR_DCdetector.png)

### SMD
![SMD Results](figures/SMD.png)

### SMAP
![SMAP Results](figures/SMAP.png)

### MSL
![MSL Results](figures/MSL.png)

### PSM
![PSM Results](figures/PSM.png)

### SWaT
![SWaT Results](figures/SWaT.png)


## Case Studies

![Internalbleeding20](figures/fig_demonstration_Internalbleeding20_time-course.svg)

![Internalbleeding19](figures/fig_demonstration_Internalbleeding19_time-course.svg)

![Internalbleeding17](figures/fig_demonstration_Internalbleeding17_time-course.svg)

![Internalbleeding15](figures/fig_demonstration_Internalbleeding15_time-course.svg)

![Internalbleeding5](figures/fig_demonstration_Internalbleeding5_time-course.svg)

![Internalbleeding14](figures/fig_demonstration_Internalbleeding14_time-course.svg)


## Input Noise Robustness Evaluation

MD-RS and MD-SW show higher noise robustness than TRAKR and RC-SRE. This indicates that the Mahalanobis distance is a more noise robust anomaly score than the reconstruction error.

### Performance for different noise strength $\sigma$
![different-sigma1](figures/fig_UCR_different-small-sigma.svg)

![different-sigma2](figures/fig_UCR_different-small-sigma_appendix.svg)

### Performance change rate for different noise strength $\sigma$ (baseline: $\sigma=0$)
![different-sigma_change-rate1](figures/fig_UCR_different-small-sigma_change-rate.svg)

![different-sigma_change-rate2](figures/fig_UCR_different-small-sigma_change-rate_appendix.svg)
