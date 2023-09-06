import ROOT
import narf.clingutils

# tensorflow libraries may have already been loaded, e.g. by the tensorflow python package
# otherwise load them explicitly, since relying on the auto loader can cause a crash if they
# are loaded by other means in the meantime (loading the tensorflow libraries twice in general
# results in a crash from duplicate registration of operations)

libs = ROOT.gInterpreter.GetSharedLibs()

if "libtensorflow_framework.so" not in libs:
    ROOT.gInterpreter.Load("libtensorflow_framework.so")

if "libtensorflow_cc.so" not in libs:
    ROOT.gInterpreter.Load("libtensorflow_cc.so")

narf.clingutils.Declare('#include "tfliteutils.h"')

