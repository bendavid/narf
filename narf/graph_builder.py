import ROOT
from .lumitools import make_lumihelper, make_jsonhelper

def build_and_run(datasets, build_function, output_file):
    results = []
    hweights = []
    evtcounts = []
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
            df = df.Filter(jsonhelper, ["run", "luminosityBlock"], "jsonhelper")

        evtcount = df.Count()

        res, hweight = build_function(df, dataset)

        results.append(res)
        hweights.append(hweight)
        evtcounts.append(evtcount)

    if lumisums:
        print("begin lumi loop")
        ROOT.ROOT.RDF.RunGraphs(lumisums.values())
        print("end lumi loop")

    for name, val in lumisums.items():
        print(name, val.GetValue())

    print("begin event loop")
    ROOT.ROOT.RDF.RunGraphs(evtcounts)
    print("done event loop")
    #for evtcount in evtcounts:
        #evtcount.GetValue()

    print(results)

    f = ROOT.TFile.Open(output_file, "RECREATE")
    for dataset, res, hweight, evtcount in zip (datasets, results, hweights, evtcounts):

        if hweight.GetEntries() != evtcount.GetValue():
            errmsg = f"Number of events for dataset {dataset.name} used to fill weight statistics {hweight.GetEntries()} not consistent with total number of events processed (after lumi filtering if applicable): {evtcount.GetValue()}"
            raise ValueError(errmsg)

        folder = f.mkdir(dataset.name)
        folder.cd()

        scaleweight = None
        if dataset.name in lumisums:
            hlumi = ROOT.TH1D("lumi", "lumi", 1, 0.5, 1.5)
            lumi = lumisums[dataset.name].GetValue()
            hlumi.Fill(1.0, lumi)
            hlumi.Write()
        hweight.Write()

        for r in res:
            print(type(r.GetValue()))
            if isinstance(r.GetValue(), ROOT.TObject):
              r.Write()
            else:
              #print("access test", r.rank())
              #print("access test", r.GetValue()[1,1,1])
              #print("access test direct", r[1,1,1])
              print("sum", r.sum())
              print("sum with overflow", r.sum(flow=True))

              #folder.WriteObject(r.GetValue(), r.name)

    f.Close()

