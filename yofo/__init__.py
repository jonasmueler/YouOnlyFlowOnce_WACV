"""YOFO: calibrated real-time radar pose estimation with multi-hypothesis normalizing flows.

Code release for "You Only Flow Once: Calibrated and Real-Time Radar Pose Estimation
with Multi-Hypothesis Normalizing Flows" (WACV 2027). The method MH-NFPG consists of a
spatiotemporal transformer backbone with a Gaussian prediction head (the MG-Prior) and a
conditional RealNVP normalizing flow (NFPG) trained on the backbone's features.
"""

__version__ = "1.0.0"
