import ROOT
import narf.tests

res = ROOT.narf.testQuantileHelperStatic()
print(f"testQuantileHelperStatic: {res}")
assert res, "testQuantileHelperStatic failed"
