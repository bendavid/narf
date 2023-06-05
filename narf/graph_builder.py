import ROOT
from .lumitools import make_lumihelper, make_jsonhelper
from .ioutils import H5PickleProxy
import time
import uuid
import sys

def build_and_run(datasets, build_function, lumi_tree = "LuminosityBlocks", event_tree = "Events", run_col = "run", lumi_col = "luminosityBlock",
        scale_xsc_lumi=False, 
        groups_to_aggregate=[]
    ):

    # TODO make this check more robust and move it to a more appropriate place
    if hasattr(ROOT, "Eigen") and ("tensorflow" in sys.modules or "jax" in sys.modules):
        raise RuntimeError("Use of Eigen headers in cling is incompatible with tensorflow and jax because of haphazard exports of possibly-conflicting Eigen symbols.  Avoid the conflict by avoiding use of Eigen or avoiding importing of tensorflow or jax.")

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
            chain = ROOT.TChain(lumi_tree)
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
                lumidf = lumidf.Filter(jsonhelper, [run_col, lumi_col], "jsonfilter")
            print("define")
            lumidf = lumidf.Define("lumival", lumihelper, [run_col, lumi_col])
            print("sum")
            lumisum = lumidf.Sum("lumival")
            print("add to dict")
            lumisums[dataset.name] = lumisum

        print("event chain")
        chain = ROOT.TChain(event_tree)
        for fpath in dataset.filepaths:
            chain.Add(fpath)

        # black magic why this needs to be protected from gc
        chains.append(chain)

        print("event df")
        df = ROOT.ROOT.RDataFrame(chain)
        if jsonhelper is not None:
            df = df.Filter(jsonhelper, [run_col, lumi_col], "jsonhelper")

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

    output_group = {}
    dsetresult_group = {}

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
        
        if scale_xsc_lumi and not dataset.is_data:
            scale = dataset.xsec * lumisums["dataPostVFP"].GetValue()*1000 / hweight[0].GetValue()
        else:
            scale = 1

        dsetresult["weight_sum"] = float(hweight[0].GetValue() * scale)
        dsetresult["event_count"] = float(evtcount.GetValue())

        if dataset.name in lumisums:
            hlumi = ROOT.TH1D("lumi", "lumi", 1, 0.5, 1.5)
            lumi = lumisums[dataset.name].GetValue()
            dsetresult["lumi"] = lumi

        if scale_xsc_lumi and dataset.group in groups_to_aggregate:
            # only sum up members if they have been scaled before
            aggregate = True

            if dataset.group not in output_group.keys():
                output_group[dataset.group] = {}

                dsetresult_group[dataset.group] = {}
                dsetresult_group[dataset.group]["dataset"] = {
                    "name": dataset.group,
                    "filepaths": dataset.filepaths,
                }
                dsetresult_group[dataset.group]["n_members"] = 1
                dsetresult_group[dataset.group]["weight_sum"] = float(hweight[0].GetValue() * scale)
                dsetresult_group[dataset.group]["event_count"] = float(evtcount.GetValue())
            else:
                dsetresult_group[dataset.group]["dataset"]["filepaths"] += dataset.filepaths
                dsetresult_group[dataset.group]["n_members"] += 1
                dsetresult_group[dataset.group]["weight_sum"] += float(hweight[0].GetValue() * scale)
                dsetresult_group[dataset.group]["event_count"] += float(evtcount.GetValue())

        else:
            aggregate = False
            output = {}
        
        for r in res:
            if isinstance(r.GetValue(), ROOT.TNamed):
                name = r.GetName()
            elif hasattr(r.GetValue(), "name"):
                name = r.GetValue().name
            else:
                name = str(uuid.uuid1())

            if scale != 1:
                histogram = scale * r.GetValue()
            else: 
                histogram = r.GetValue()

            if aggregate:
                if name not in output_group[dataset.group]:
                    output_group[dataset.group][name] = [histogram,]
                else:
                    output_group[dataset.group][name].append(histogram)
            else:
                if name in output.keys():
                    print(f"Warning: the histogram {name} is defined more than once! It will be overwritten.")
                    
                output[name] = H5PickleProxy(histogram)

        if not aggregate:

            dsetresult["output"] = output

            resultdict[dataset.name] = dsetresult

    for g_name, group in output_group.items():
        output = {}
        print(f"--- Aggregate group {g_name}")
        for h_name, histograms in group.items():

            if len(histograms) != dsetresult_group[g_name]["n_members"]:
                print(f"Warning: for {h_name} from group {g_name} there is a different number of histograms ({len(histograms)}) than original members ("+str(dsetresult_group[g_name]["n_members"])+")")
                print("   Summing them up probably leads to wrong behavious")
                # FIXME: Currently, if a histogram is only available in some members but not in outers of the same group it is still added up,
                #   which is probably not desired.

            output[h_name] = H5PickleProxy(sum(histograms))
            
        dsetresult = dsetresult_group[g_name]

        dsetresult["output"] = output

        resultdict[g_name] = dsetresult

    time_done = time.time()

    print("narf build graphs:", time_built - time0)
    print("narf lumi loop:", time_done_lumi - time_built)
    print("narf event loop:", time_done_event - time_done_lumi)
    print("narf build results:", time_done - time_done_event)

    return resultdict
