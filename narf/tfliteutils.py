import ROOT
import narf.clingutils

# tensorflow libraries may have already been loaded, e.g. by the tensorflow python package
# otherwise load them explicitly, since relying on the auto loader can cause a crash if they
# are loaded by other means in the meantime (loading the tensorflow libraries twice in general
# results in a crash from duplicate registration of operations)

# try to load tensorflow lite libraries first in case they are present

loaded_lite = ROOT.gInterpreter.Load("libtensorflowlite.so") == 0
loaded_lite_flex = ROOT.gInterpreter.Load("libtensorflowlite_flex.so") == 0

if loaded_lite != loaded_lite_flex:
    raise RuntimeError("Inconsistent installation of Tensorflow Lite detected, must have both libtensorflowlite and libtensorflowlite_flex (or neither of them)")

if not loaded_lite and "libtensorflow_framework.so" not in ROOT.gInterpreter.GetSharedLibs():
    ret = ROOT.gInterpreter.Load("libtensorflow_framework.so")
    if ret != 0:
        raise RuntimeError("Failed to load libtensorflow_framework.so")

if not loaded_lite and "libtensorflow_cc.so" not in ROOT.gInterpreter.GetSharedLibs():
    ret = ROOT.gInterpreter.Load("libtensorflow_cc.so")
    if ret != 0:
        raise RuntimeError("Failed to load libtensorflow_cc.so")

narf.clingutils.Load(f"{narf.common.base_dir}/narf/lib/libnarf.so")

# ROOT.gInterpreter.AddIncludePath(f"/usr/include/tensorflow")
# ROOT.gInterpreter.AddIncludePath(f"/usr/include/eigen3")

narf.clingutils.Declare('#include "tfliteutils.h"')
narf.clingutils.Declare('#include "tfccutils.h"')
# narf.clingutils.Declare('#include "tfcutils.h"')
#
# print("testing tf c interface")
# #
# #
# model = "/scratch/submit/cms/jbendavi/wmassdev49/WRemnants/wremnants-data/data/calibration/muon_response"
#
#
#
# input("wait pre")
#
# # helpers = []
#
# # for i in range(128):
#     # helper = ROOT.narf.tf_session(model)
#     # helpers.append(helper)
#
# helper = ROOT.narf.tf_session(model)

# input("wait post")



# ROOT.gSystem.AddIncludePath("/usr/include/tensorflow")

# ROOT.gROOT.ProcessLine(f".L {narf.common.base_dir}/narf/src/tfccutils.cpp+")

# ROOT.gSystem.CompileMacro(f"{narf.common.base_dir}/narf/src/tfccutils.cpp")

#Singularity> g++ -std=c++17 -O3 -march=x86-64-v3 -shared -fpic -I/usr/include/tensorflow -Ltensorflow-cc -o testtfcc.so testtfcc.cpp

# print("testing tf cc interface")
#
# input("wait pre")
#
# helper = ROOT.narf.tf_helper(model, "serving_default", 128)
#
# # helpers = []
# # for i in range(128):
# #     helper = ROOT.narf.tf_model_data(model, "serving_default")
# #     helpers.append(helper)
#
# input("wait post")
