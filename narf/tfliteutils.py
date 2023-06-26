import ROOT
import narf.clingutils

ROOT.gInterpreter.Load("libtensorflowlite.so")
ROOT.gInterpreter.Load("libtensorflowlite_flex.so")

narf.clingutils.Declare('#include "tfliteutils.h"')
