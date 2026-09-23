# Systematic Ablation Study: MetaWorld
- **Rollout Horizon**: 20
- **Evaluated Held-Out Trajectories**: 40
- **Date**: 2026-09-23 02:36:06

## 1. Gating & Control Baseline Ablation
| Architecture Variant | Mean Horizon MSE | Final Step MSE | Gain vs Base (%) | Cohen's d | p-Value | Sig |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (JEPA-WM)** | 1.5451 | 1.8372 | +0.00% | 0.000 | 1.00e+00 | n.s. |
| **Static Residual Control** | 1.4644 | 1.8955 | +5.22% | 0.219 | 1.74e-01 | n.s. |
| **Fixed Blend ($\lambda=0.30$)** | 1.4756 | 1.8942 | +4.50% | 0.191 | 2.35e-01 | n.s. |
| **Horizon Decay ($\gamma=0.95$)** | 1.5137 | 1.9127 | +2.03% | 0.085 | 5.96e-01 | n.s. |
| **Adaptive Confidence Gating** | 1.5529 | 1.8918 | -0.50% | -0.020 | 9.01e-01 | n.s. |
| **Full RAG-WM (Adaptive + Decay)** | 1.5662 | 1.8963 | -1.37% | -0.053 | 7.37e-01 | n.s. |

## 2. Retrieval Neighborhood Size (K) Sensitivity
| Neighborhood Size K | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|
| K=0 | 1.5451 | 1.8372 |
| K=1 | 1.4853 | 1.9099 |
| K=2 | 1.4775 | 1.9004 |
| K=3 | 1.4756 | 1.8942 |
| K=5 | 1.4708 | 1.8899 |
| K=8 | 1.4716 | 1.8924 |
| K=10 | 1.4723 | 1.8973 |

## 3. Blend Weight (λ) Sensitivity
| Interpolation Weight λ | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|
| λ=0.00 | 1.5451 | 1.8372 |
| λ=0.10 | 1.5784 | 1.8858 |
| λ=0.20 | 1.5391 | 1.9008 |
| λ=0.30 | 1.4756 | 1.8942 |
| λ=0.40 | 1.3917 | 1.8904 |
| λ=0.50 | 1.2834 | 1.8849 |
| λ=0.70 | 0.9302 | 1.7321 |
| λ=1.00 | 1.1067 | 1.9486 |

## 4. Memory Bank Capacity Scaling
| Experience Ratio | Transitions Indexed | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|:---:|
| 10% | 784 | 1.4726 | 1.8976 |
| 25% | 1960 | 1.4756 | 1.9014 |
| 50% | 3920 | 1.4736 | 1.8910 |
| 75% | 5880 | 1.4737 | 1.8964 |
| 100% | 7840 | 1.4756 | 1.8942 |

## 5. Softmax Temperature (τ) Sensitivity
| Softmax Temperature τ | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|
| τ=0.02 | 1.4757 | 1.8943 |
| τ=0.05 | 1.4756 | 1.8942 |
| τ=0.10 | 1.4756 | 1.8942 |
| τ=0.20 | 1.4755 | 1.8938 |
| τ=0.50 | 1.4756 | 1.8939 |
| τ=1.00 | 1.4756 | 1.8939 |
