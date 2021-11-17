import ROOT
#ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()

import narf
from datasets import datasets2016

#import gc

datasets = [data, zmc]
#datasets = [zmc, data]
#datasets = [zmc]

def build_graph(df, dataset):
    df = df.Filter("nMuon > 0")
    df = df.Define("Muon0_pt", "Muon_pt[0]")
    df = df.Define("Muon0_eta", "Muon_eta[0]")

    if dataset.is_data:
        df = df.Define("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.Define("one", "1.0")

    hweight = df.Histo1D(("sumweights", "", 1, 0.5, 1.5), "one", "weight")

    hpt = df.Histo1D(("hpt", "", 35, 25., 60.), "Muon0_pt", "weight")
    heta = df.Histo1D(("heta", "", 48, -2.4, 2.4), "Muon0_eta", "weight")

    #hpt.GetResult()
    results = [hpt, heta]
    #results = [hpt]

    return results, hweight

narf.build_and_run(datasets, build_graph, "testout.root")
