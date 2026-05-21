import ROOT
import narf.tests

res = ROOT.narf.testMapWrapper()
print(f"testMapWrapper: {res}")
assert res, "testMapWrapper failed"
