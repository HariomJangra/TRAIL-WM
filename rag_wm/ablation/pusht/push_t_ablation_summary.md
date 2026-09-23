# Systematic Ablation Study: Push-T
- **Rollout Horizon**: 20
- **Evaluated Held-Out Trajectories**: 21
- **Date**: 2026-09-23 02:48:52

## 1. Gating & Control Baseline Ablation
| Architecture Variant | Mean Horizon MSE | Final Step MSE | Gain vs Base (%) | Cohen's d | p-Value | Sig |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (JEPA-WM)** | 3.8795 | 4.2504 | +0.00% | 0.000 | 1.00e+00 | n.s. |
| **Static Residual Control** | 3.7137 | 4.2618 | +4.27% | 0.805 | 1.45e-03 | ** |
| **Fixed Blend ($\lambda=0.30$)** | 3.6926 | 4.2500 | +4.82% | 0.804 | 1.47e-03 | ** |
| **Horizon Decay ($\gamma=0.95$)** | 3.7361 | 4.2873 | +3.69% | 0.605 | 1.18e-02 | * |
| **Adaptive Confidence Gating** | 3.8535 | 4.2699 | +0.67% | 0.202 | 3.65e-01 | n.s. |
| **Full TRAIL-WM (Adaptive + Decay)** | 3.8535 | 4.2699 | +0.67% | 0.202 | 3.65e-01 | n.s. |

## 2. Retrieval Neighborhood Size (K) Sensitivity
| Neighborhood Size K | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|
| K=0 | 3.8795 | 4.2504 |
| K=1 | 3.6903 | 4.2437 |
| K=2 | 3.6918 | 4.2476 |
| K=3 | 3.6926 | 4.2500 |
| K=5 | 3.6913 | 4.2444 |
| K=8 | 3.6912 | 4.2438 |
| K=10 | 3.6923 | 4.2468 |

## 3. Blend Weight (λ) Sensitivity
| Interpolation Weight λ | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|
| λ=0.00 | 3.8795 | 4.2504 |
| λ=0.10 | 3.8331 | 4.2328 |
| λ=0.20 | 3.7714 | 4.2436 |
| λ=0.30 | 3.6926 | 4.2500 |
| λ=0.40 | 3.5970 | 4.2838 |
| λ=0.50 | 3.4590 | 4.2448 |
| λ=0.70 | 2.9712 | 4.1300 |
| λ=1.00 | 0.8572 | 1.5644 |

## 4. Memory Bank Capacity Scaling
| Experience Ratio | Transitions Indexed | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|:---:|
| 10% | 540 | 3.6989 | 4.2513 |
| 25% | 1350 | 3.6956 | 4.2511 |
| 50% | 2700 | 3.6957 | 4.2675 |
| 75% | 4050 | 3.6931 | 4.2584 |
| 100% | 5400 | 3.6926 | 4.2500 |

## 5. Softmax Temperature (τ) Sensitivity
| Softmax Temperature τ | Mean Horizon MSE | Final Step MSE |
|:---:|:---:|:---:|
| τ=0.02 | 3.6924 | 4.2497 |
| τ=0.05 | 3.6926 | 4.2499 |
| τ=0.10 | 3.6926 | 4.2500 |
| τ=0.20 | 3.6926 | 4.2500 |
| τ=0.50 | 3.6926 | 4.2500 |
| τ=1.00 | 3.6926 | 4.2500 |
