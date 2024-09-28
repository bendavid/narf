import ROOT

def Declare(code):
    ret = ROOT.gInterpreter.Declare(code)

    if not ret:
        raise ValueError(f"Call to gInterpreter.Declare failed")

def Load(filename):
    ret = ROOT.gInterpreter.Load(filename)

    if ret != 0:
        raise ValueError(f"Call to gInterpreter.Load failed")
