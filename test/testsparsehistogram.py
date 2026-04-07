import ROOT
import numpy as np
import boost_histogram as bh
import narf
import narf.histutils
from wums.sparse_hist import SparseHist

ROOT.ROOT.EnableImplicitMT(4)

N = 20000
df = ROOT.RDataFrame(N)
df = df.Define("x", "double((rdfentry_ % 20) + 0.5)")  # values 0.5, 1.5, ..., 19.5
df = df.Define("w", "1.0")

ax = bh.axis.Regular(20, 0.0, 20.0)
res = df.HistoBoost("hsparse", [ax], ["x", "w"], storage=narf.histutils.SparseStorage(fill_fraction=1.0))
sh = res.GetValue()

assert isinstance(sh, SparseHist), f"expected SparseHist, got {type(sh).__name__}"
assert sh.nnz == 20, f"expected 20 populated bins, got {sh.nnz}"

# Dense round-trip with flow gives a (22,) array; with flow=False a (20,) array.
dense_noflow = sh.toarray(flow=False)
expected = np.full(20, N // 20, dtype=np.float64)
assert np.allclose(dense_noflow, expected), f"mismatch: {dense_noflow} vs {expected}"


# ---- 3D fill: verify ND linearization ----
N3 = 24000
df3 = ROOT.RDataFrame(N3)
df3 = df3.Define("x", "double(rdfentry_ % 3)")        # 0, 1, 2
df3 = df3.Define("y", "double((rdfentry_ / 3) % 4)")  # 0..3
df3 = df3.Define("z", "double((rdfentry_ / 12) % 5)") # 0..4
df3 = df3.Define("w", "1.0")

ax_x = bh.axis.Regular(3, 0.0, 3.0, underflow=False, overflow=False)
ax_y = bh.axis.Regular(4, 0.0, 4.0, underflow=False, overflow=False)
ax_z = bh.axis.Regular(5, 0.0, 5.0, underflow=False, overflow=False)

res3 = df3.HistoBoost(
    "h3d",
    [ax_x, ax_y, ax_z],
    ["x", "y", "z", "w"],
    storage=narf.histutils.SparseStorage(fill_fraction=1.0),
)
sh3 = res3.GetValue()

assert isinstance(sh3, SparseHist)
arr3 = sh3.toarray(flow=False)
assert arr3.shape == (3, 4, 5)
# Each (x,y,z) cell receives N3 / (3*4*5) = 400 entries.
assert np.allclose(arr3, np.full((3, 4, 5), N3 // (3 * 4 * 5)))

# Cross-check the same data through a dense fill.
df3_dense = ROOT.RDataFrame(N3)
df3_dense = df3_dense.Define("x", "double(rdfentry_ % 3)")
df3_dense = df3_dense.Define("y", "double((rdfentry_ / 3) % 4)")
df3_dense = df3_dense.Define("z", "double((rdfentry_ / 12) % 5)")
df3_dense = df3_dense.Define("w", "1.0")
hd = df3_dense.HistoBoost("h3d_dense", [ax_x, ax_y, ax_z], ["x", "y", "z", "w"]).GetValue()
assert np.allclose(arr3, hd.values())

print("testSparseHistogram OK")
