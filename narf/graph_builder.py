import ROOT
from .lumitools import make_lumihelper, make_jsonhelper
from wums.ioutils import H5PickleProxy
import time
import uuid
import sys
import subprocess
import shlex
import narf.rdfutils

def build_and_run(datasets, build_function, lumi_tree = "LuminosityBlocks", event_tree = "Events", run_col = "run", lumi_col = "luminosityBlock"):

    # TODO make this check more robust and move it to a more appropriate place
    if hasattr(ROOT, "Eigen"):
        libs = ROOT.gInterpreter.GetSharedLibs().split()
        check_symbols = []
        check_symbols.append(b"Eigen::internal::TensorBlockScratchAllocator<Eigen::DefaultDevice>::allocate")
        check_symbols.append(b"Eigen::TensorEvaluator<Eigen::TensorSlicingOp<Eigen::DSizes<long, 8> const, Eigen::DSizes<long, 8> const, Eigen::TensorMap<Eigen::Tensor<signed char, 8, 1, long>, 0, Eigen::MakePointer> > const, Eigen::DefaultDevice>::TensorEvaluator(Eigen::TensorSlicingOp<Eigen::DSizes<long, 8> const, Eigen::DSizes<long, 8> const, Eigen::TensorMap<Eigen::Tensor<signed char, 8, 1, long>, 0, Eigen::MakePointer> > const&, Eigen::DefaultDevice const&)")

        for lib in libs:
            if "libtensorflow_framework.so" in lib or "libtensorflow_cc.so" in lib:
                proc = subprocess.Popen(shlex.split(f"nm -gDC --just-symbols {lib}"), stdout=subprocess.PIPE)

                for symbol in proc.stdout:
                    for check_symbol in check_symbols:
                        if check_symbol in symbol:
                            raise RuntimeError("Tensorflow has been loaded simultaneously with jitted Eigen functions, but the Tensorflow libraries contain conflicting symbols.  Use a Tensorflow build which fixes this problem, or avoid loading the Tensorflow libraries, e.g. from importing the tensorflow python package.")

                try:
                    ret = proc.wait(5)
                except:
                    proc.kill()
                    raise TimeoutError(f"Failed to check symbols in library {lib}")

                if ret != 0:
                    raise RuntimeError(f"Failed to check symbols in library {lib}: Return code: {ret}")

    time0 = time.time()

    dfs = []
    results = []
    hweights = []
    evtcounts = []
    chains = []
    lumidfs = []
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
            lumidfs.append(lumidf)
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
        dfs.append(df)
        if jsonhelper is not None:
            df = df.Filter(jsonhelper, [run_col, lumi_col], "jsonhelper")

        evtcount = df.Count()

        res, hweight = build_function(df, dataset)

        results.append(res)
        hweights.append(hweight)
        evtcounts.append(evtcount)

    time_built = time.time()

    # short interval for progress bar for interactive use
    # longer interval otherwise to avoid blowing up logs
    if sys.stdout.isatty():
        interval = 1
    else:
        interval = 5*60

    if lumidfs:
        # need to flush before the event loop to keep output consistently ordered
        print("begin lumi loop", flush = True)
        # call every entry for lumi tree otherwise not enough statistics for
        # printouts every 1s usually
        ROOT.narf.RunGraphsWithProgressBar(lumidfs, 1, interval)
        print("end lumi loop")
    time_done_lumi = time.time()

    for name, val in lumisums.items():
        print(name, val.GetValue())

    print("begin event loop", flush = True)
    ROOT.narf.RunGraphsWithProgressBar(dfs, 1000, interval)
    print("done event loop")
    time_done_event = time.time()

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
                name = r.GetName()
            elif hasattr(r.GetValue(), "name"):
                name = r.GetValue().name
            else:
                name = str(uuid.uuid1())

            output[name] = H5PickleProxy(r.GetValue())

        dsetresult["output"] = output

        resultdict[dataset.name] = dsetresult

    time_done = time.time()

    print("narf build graphs:", time_built - time0)
    print("narf lumi loop:", time_done_lumi - time_built)
    print("narf event loop:", time_done_event - time_done_lumi)
    print("narf build results:", time_done - time_done_event)

    return resultdict
