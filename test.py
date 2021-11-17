import ROOT
#ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()

import narf

#import gc

lumicsv = "data/bylsoutput.csv"
lumijson = "data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

data = narf.Dataset(name = "dataPostVFP",
                    filepaths = ["root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/SingleMuon/NanoV8Data/Run2016G_210302_203023/0000/NanoV8Data_100.root"],
                    is_data = True,
                    lumi_csv = lumicsv,
                    lumi_json = lumijson)

zmc = narf.Dataset(name = "ZmumuPostVFP",
                  filepaths = ["root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0000/NanoV8MCPostVFP_weightFix_100.root"
                    #"root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/    DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0000/*.root",
                    #"root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0001/*.root",
                  ],
                  is_data = False,
                  target_data = "dataPostVFP")

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

    hweight = df.Histo1D(("hweight", "", 1, 0.5, 1.5), "one", "weight")

    hpt = df.Histo1D(("hpt", "", 35, 25., 60.), "Muon0_pt", "weight")
    heta = df.Histo1D(("heta", "", 48, -2.4, 2.4), "Muon0_eta", "weight")

    #hpt.GetResult()
    results = [hpt, heta]
    #results = [hpt]

    return results, hweight

narf.build_and_run(datasets, build_graph, "testout.root")
