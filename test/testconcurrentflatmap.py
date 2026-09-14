import ROOT
import narf.tests

res = ROOT.narf.testConcurrentFlatMap()
print(f"testConcurrentFlatMap: {res}")
assert res, "testConcurrentFlatMap failed"
