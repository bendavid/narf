import ROOT
import boost_histogram as bh
import narf
import narf.histutils

ROOT.ROOT.EnableImplicitMT(4)

N = 20000
df = ROOT.RDataFrame(N)
df = df.Define("x", "double((rdfentry_ % 20) + 0.5)")  # values 0.5, 1.5, ..., 19.5
df = df.Define("w", "1.0")

ax = bh.axis.Regular(20, 0.0, 20.0)
res = df.HistoBoost("hsparse", [ax], ["x", "w"], storage=narf.histutils.SparseStorage(fill_fraction=1.0))
hfill = res.GetValue()

expected_per_bin = N // 20

# Snapshot populated bins (linearized index, value).
got = {int(k): float(v) for k, v in ROOT.narf.sparse_histogram_snapshot(hfill)}

print(f"populated bins: {len(got)}")
assert len(got) == 20, f"expected 20 populated bins, got {len(got)}"

# boost::histogram linearized layout for a single regular axis with under/overflow:
#   linearized = boost_index + 1, real bins occupy keys 1..20.
for b in range(1, 21):
    assert b in got, f"missing bin {b}"
    assert abs(got[b] - expected_per_bin) < 1e-9, f"bin {b}: {got[b]} != {expected_per_bin}"

print("testSparseHistogram OK")
