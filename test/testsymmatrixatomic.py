import ROOT
import narf.tests

res = ROOT.narf.testSymMatrixAtomic()
print(f"testSymMatrixAtomic: {res}")
assert res, "testSymMatrixAtomic failed"
