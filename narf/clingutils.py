import ROOT

def Declare(code):
    ret = ROOT.gInterpreter.Declare(code)

    if not ret:
        raise ValueError("Call to gInterpreter.Declare failed")

def Load(lib):
    ret = ROOT.gInterpreter.Load(lib)

    if ret == -1:
        raise ValueError(f"Call to gInterpreter.Load({lib}) failed")
