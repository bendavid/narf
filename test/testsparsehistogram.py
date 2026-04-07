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

print("testSparseHistogram OK")
