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

narf.clingutils.Declare('#include "tfliteutils.h"')

