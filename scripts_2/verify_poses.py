# verify_poses.py
import json, numpy as np, sys
from numpy.linalg import norm, det

poses = json.load(open(sys.argv[1]))
for i, p in enumerate(poses):
    T = np.array(p["T_board_cam"])
    R = T[:3,:3]
    # orthonormal check
    ortho_err = norm(R @ R.T - np.eye(3))
    detR = det(R)
    t = T[:3,3]
    print(f"[{i}] ortho_err={ortho_err:.3e}, det(R)={detR:.3f}, t={t}")
