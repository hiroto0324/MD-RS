# MD-RS (Under Review)

## Architecture
![schematic](figures/fig_schematic_MD-RS.svg)

## Overview
![demo](figures/fig_demonstration_MD-RS.svg)

# Get Started

1. Install Python ..., PyTorch >= ....
2. Download data. You can obtain all benchmarks from [Google Cloud](). All the datasets are well pre-processed.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder  `./scripts`. You can reproduce the experiment results as follows:


# Main Results

All of these results were calculated **without point adjustment (PA)**.

## Overall Results

### UCR
![UCR Results](figures/UCR.png)

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


## Noise Robustness Evaluation

### Performance for different noise strength $\sigma$
![different-sigma1](figures/fig_UCR_different-small-sigma.svg)

![different-sigma2](figures/fig_UCR_different-small-sigma_appendix.svg)

### Performance change rate for different noise strength $\sigma$ (baseline: $\sigma=0$)
![different-sigma_change-rate1](figures/fig_UCR_different-small-sigma_change-rate.svg)

![different-sigma_change-rate2](figures/fig_UCR_different-small-sigma_change-rate_appendix.svg)
