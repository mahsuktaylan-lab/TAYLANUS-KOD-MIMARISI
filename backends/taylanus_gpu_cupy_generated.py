"""
TAYLANUS v2.23 generated CuPy backend skeleton.
GENERATED ONLY — not executed/validated in the current environment.

Expected runtime:
    CuPy + cupyx.scipy.sparse + CUDA-capable GPU
"""
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_linalg

class TaylanusCuPyBackend:
    name = "GPU_CUPY"

    def __init__(self, cpu_ir):
        self.cpu_ir = cpu_ir

    @staticmethod
    def to_gpu_csr(M):
        return cpx_sparse.csr_matrix(
            (cp.asarray(M.data), cp.asarray(M.indices), cp.asarray(M.indptr)),
            shape=M.shape
        )

    def lower_subface_ir(self, sub_ir):
        out = {"D": [[None]*3 for _ in range(3)],
               "LAP": [None]*3,
               "ADV": [[None]*3 for _ in range(3)]}
        for a in range(3):
            for j in range(3):
                out["D"][a][j] = self.to_gpu_csr(sub_ir["D"][a][j])
            out["LAP"][a] = self.to_gpu_csr(sub_ir["LAP"][a])
            for b in range(3):
                out["ADV"][a][b] = self.to_gpu_csr(sub_ir["ADV"][a][b])
        return out

    def subface_rhs(self, face_u, ir, nu):
        out=[]
        for a in range(3):
            d=[ir["D"][a][j].dot(face_u[a]) for j in range(3)]
            av=[ir["ADV"][a][b].dot(face_u[b]) for b in range(3)]
            lap=ir["LAP"][a].dot(face_u[a])
            out.append(-(av[0]*d[0]+av[1]*d[1]+av[2]*d[2]) + nu*lap)
        return out

# v2.23 intentionally leaves pressure-factorization and topology transfer
# backend hooks explicit. They must be validated on real CUDA hardware.
