import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--useBoost", type=bool, help="user boost histograms", default=False)
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if args.nThreads is not None:
    if args.nThreads > 1:
        ROOT.ROOT.EnableImplicitMT(args.nThreads)
else:
    ROOT.ROOT.EnableImplicitMT()

#ROOT.TTreeProcessorMT.SetTasksPerWorkerHint(1)

import pickle
import gzip


import narf
from datasets import datasets2016
import hist
import lz4.frame

datasets = datasets2016.allDatasets()

boost_hist_default = ROOT.boost.histogram.use_default
boost_hist_options_none = ROOT.boost.histogram.axis.option.none_t

# standard regular axes

axis_pt = hist.axis.Regular(29, 26., 55., name = "pt")
axis_eta = hist.axis.Regular(48, -2.4, 2.4, name = "eta")

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

# integer axis with no overflow
axis_pdf_idx = hist.axis.Integer(0, 103, underflow=False, overflow=False, name = "pdfidx")


def build_graph(df, dataset):
    results = []

    if dataset.is_data:
        df = df.Define("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05")

    df = df.Filter("Sum(vetoMuons) == 1")

    df = df.Define("goodMuons", "vetoMuons && Muon_mediumId")

    df = df.Filter("Sum(goodMuons) == 1")

    df = df.Define("goodMuons_Pt0", "Muon_pt[goodMuons][0]")
    df = df.Define("goodMuons_Eta0", "Muon_eta[goodMuons][0]")
    df = df.Define("goodMuons_Charge0", "Muon_charge[goodMuons][0]")

    if args.useBoost:
        #hptetacharge = df.HistoBoost("hptetacharge", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight"])
        hptetacharge = df.HistoBoost("hptetacharge", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight"])
        #hptetacharge = df.HistoBoost("hptetacharge", [axis_pt], ["goodMuons_Pt0", "weight"])
    else:
        #hptetacharge = df.Histo3D(("hptetacharge", "", 29, 26., 55., 48, -2.4, 2.4, 2, -2., 2.), "goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight")
        hptetacharge = df.Histo3DWithBoost(("hptetacharge", "", 29, 26., 55., 48, -2.4, 2.4, 2, -2., 2.), "goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight")
        #hptetacharge = df.Histo1DWithBoost(("hptetacharge", "", 29, 26., 55.), "goodMuons_Pt0", "weight")
    results.append(hptetacharge)

    if not dataset.is_data:

        #df = df.Define("pdfweight", "weight*LHEPdfWeight")

        #df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")

        #hptetachargepdf = df.HistoBoost("hptetachargepdf", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", "pdfweight"])
        #results.append(hptetachargepdf)

        df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")


        for i in range(10):

            df = df.Define(f"pdfweight_{i}", "weight*LHEPdfWeight")

            if args.useBoost:
                hptetachargepdf = df.HistoBoost(f"hptetachargepdf_{i}", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", f"pdfweight_{i}"])
            else:
                #hptetachargepdf = df.HistoND((f"hptetachargepdf_{i}", "", 4, [29, 48, 2, 103], [26., -2.4, -2., -0.5], [55., 2.4, 2., 102.5]), ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", f"pdfweight_{i}"])
                hptetachargepdf = df.HistoNDWithBoost((f"hptetachargepdf_{i}", "", 4, [29, 48, 2, 103], [26., -2.4, -2., -0.5], [55., 2.4, 2., 102.5]), ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", f"pdfweight_{i}"])
            results.append(hptetachargepdf)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

fname = "test.pkl.lz4"

print("writing output")
#with gzip.open(fname, "wb") as f:
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f)
