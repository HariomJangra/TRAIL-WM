# Systematic Ablation Study: PointMaze
- **Rollout Horizon**: 20
- **Evaluated Held-Out Trajectories**: 4
- **Date**: 2026-09-23 02:12:59

## 1. Gating & Control Baseline Ablation
| Architecture Variant | Mean Horizon MSE | Final Step MSE | Gain vs Base (%) | Cohen's d | p-Value | Sig |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (JEPA-WM)** | 5.3549 | 5.3892 | +0.00% | 0.000 | 1.00e+00 | n.s. |
| **Static Residual Control** | 5.1078 | 5.3512 | +4.61% | 5.690 | 1.46e-03 | ** |
| **Fixed Blend ($\lambda=0.30$)** | 5.0791 | 5.3256 | +5.15% | 6.497 | 9.84e-04 | *** |
| **Horizon Decay ($\gamma=0.95$)** | 5.1034 | 5.3550 | +4.70% | 5.830 | 1.35e-03 | ** |
| **Adaptive Confidence Gating** | 5.2147 | 5.3691 | +2.62% | 4.123 | 3.73e-03 | ** |
| **Full TRAIL-WM (Adaptive + Decay)** | 5.2147 | 5.3691 | +2.62% | 4.123 | 3.73e-03 | ** |
