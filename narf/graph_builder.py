import ROOT
from .lumitools import make_lumihelper, make_jsonhelper

def build_and_run(datasets, build_function, output_file):
    results = []
    hweights = []
    chains = []
    lumisums = {}

    for dataset in datasets:
        print("start of loop")

        jsonhelper = None
        if dataset.lumi_json is not None:
           jsonhelper = make_jsonhelper(dataset.lumi_json)

        if dataset.is_data and dataset.lumi_csv is not None:
            chain = ROOT.TChain("LuminosityBlocks")
            for fpath in dataset.filepaths:
                chain.Add(fpath)
            chains.append(chain)
            print("making lumi helper")
            lumihelper = make_lumihelper(dataset.lumi_csv)
            print("making df")
            lumidf = ROOT.ROOT.RDataFrame(chain)
            if jsonhelper is not None:
                print("adding lumi filter")
                print(jsonhelper)
                lumidf = lumidf.Filter(jsonhelper, ["run", "luminosityBlock"], "jsonfilter")
            print("define")
            lumidf = lumidf.Define("lumival", lumihelper, ["run", "luminosityBlock"])
            print("sum")
            lumisum = lumidf.Sum("lumival")
            print("add to dict")
            lumisums[dataset.name] = lumisum

        print("event chain")
        chain = ROOT.TChain("Events")
        for fpath in dataset.filepaths:
            chain.Add(fpath)

        # black magic why this needs to be protected from gc
        chains.append(chain)

        print("event df")
        df = ROOT.ROOT.RDataFrame(chain)
        if jsonhelper is not None:
            lumidf = lumidf.Filter(jsonhelper, ["run", "luminosityBlock"], "jsonhelper")

        res, hweight = build_function(df, dataset)

        results.append(res)
        hweights.append(hweight)

    if lumisums:
        ROOT.ROOT.RDF.RunGraphs(lumisums.values())

    for name, val in lumisums.items():
        print(name, val.GetValue())

    ROOT.ROOT.RDF.RunGraphs(hweights)

    print(results)

    for dataset, res, hweight in zip (datasets, results, hweights):
        lumi = None
        if dataset.target_lumi is not None:
            lumi = dataset.target_lumi
        elif dataset.target_data is not None:
            lumi = lumisums[dataset.target_data].GetValue()

        scaleweight = None
        if lumi is not None and dataset.xsec is not None:
            scaleweight = lumi*xsec/hweight.GetSumOfWeights()
        for r in res:
            if hasattr(r, "GetName") and hasattr(r, "SetName"):
                r.SetName(f"{r.GetName()}_{dataset.name}")
            if hasattr(r, "Scale") and scaleweight is not None:
                print("scaling", r.GetName())
                r.Scale(scaleweight)


    f = ROOT.TFile.Open(output_file, "RECREATE")
    f.cd()
    for res in results:
        for r in res:
            print("writing")
            r.Write()
    f.Close()

