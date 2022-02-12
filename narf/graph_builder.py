import ROOT
from .lumitools import make_lumihelper, make_jsonhelper
import time

def build_and_run(datasets, build_function):
    time0 = time.time()

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

    time_built = time.time()


    if lumisums:
        print("begin lumi loop")
        ROOT.ROOT.RDF.RunGraphs(lumisums.values())
        print("end lumi loop")
    time_done_lumi = time.time()

    for name, val in lumisums.items():
        print(name, val.GetValue())

    print("begin event loop")
    ROOT.ROOT.RDF.RunGraphs(evtcounts)
    print("done event loop")
    time_done_event = time.time()

    #print(results)

    resultdict = {}

    for dataset, res, hweight, evtcount in zip (datasets, results, hweights, evtcounts):

        if hweight[1].GetValue() != evtcount.GetValue():
            errmsg = f"Number of events for dataset {dataset.name} used to fill weight statistics {hweight[1].GetValue()} not consistent with total number of events processed (after lumi filtering if applicable): {evtcount.GetValue()}"
            raise ValueError(errmsg)

        dsetresult = {}

        dsetresult["dataset"] = { "name" : dataset.name,
                                  "filepaths" : dataset.filepaths,
                                  "is_data" : dataset.is_data,
                                  "xsec" : dataset.xsec,
                                  "lumi_csv" : dataset.lumi_csv,
                                  "lumi_json" : dataset.lumi_json,
                                }


        dsetresult["weight_sum"] = float(hweight[0].GetValue())
        dsetresult["event_count"] = float(evtcount.GetValue())

        if dataset.name in lumisums:
            hlumi = ROOT.TH1D("lumi", "lumi", 1, 0.5, 1.5)
            lumi = lumisums[dataset.name].GetValue()
            dsetresult["lumi"] = lumi

        output = {}

        for r in res:
            if isinstance(r.GetValue(), ROOT.TNamed):
              output[r.GetName()] = r.GetValue()
              #if isinstance(r.GetValue(), ROOT.THnBase):
                  #print(r.GetSumw())
                  #print(r.GetSumw2())

            elif hasattr(r.GetValue(), "name"):
                output[r.GetValue().name] = r.GetValue()
                print("sum", r.sum())
                print("sum with overflow", r.sum(flow=True))
            else:
                output[str(hash(r.GetValue()))] = r.GetValue()

        dsetresult["output"] = output

        resultdict[dataset.name] = dsetresult

    time_done = time.time()

    print("narf build graphs:", time_built - time0)
    print("narf lumi loop:", time_done_lumi - time_built)
    print("narf event loop:", time_done_event - time_done_lumi)
    print("narf build results:", time_done - time_done_event)

    return resultdict
