import ROOT
import narf.tests

res = ROOT.narf.testSparseMatrixAtomic()
print(f"testSparseMatrixAtomic: {res}")
assert res, "testSparseMatrixAtomic failed"
