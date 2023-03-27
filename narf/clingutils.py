import ROOT

def Declare(code):
    ret = ROOT.gInterpreter.Declare(code)

    if not ret:
        raise ValueError("Call to gInterpreter.Declare failed")
