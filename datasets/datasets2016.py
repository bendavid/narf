import narf 

lumicsv = "data/bylsoutput.csv"
lumijson = "data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

def allDatasets():
    data = narf.Dataset(name = "dataPostVFP",
                        filepaths = data_files_,
                        is_data = True,
                        lumi_csv = lumicsv,
                        lumi_json = lumijson)

    zmc = narf.Dataset(name = "ZmumuPostVFP",
                    filepaths = zmc_files_,
                    is_data = False,)
                    

    return {"data" : data, "zmc" : zmc}


data_files_ = ["root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/SingleMuon/NanoV8Data/Run2016G_210302_203023/0000/NanoV8Data_100.root"]

zmc_files_ = ["root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0000/NanoV8MCPostVFP_weightFix_100.root"
                        #"root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/    DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0000/*.root",
                        #"root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/0001/*.root",
            ]
