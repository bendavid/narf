import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT(128)

import narf
from datasets import datasets2016

datasets = datasets2016.allDatasets()

axis_pt = ROOT.boost.histogram.axis.regular[""](29, 26., 55., "pt")
axis_eta = ROOT.boost.histogram.axis.regular[""](48, -2.4, 2.4, "eta")
axis_charge = ROOT.boost.histogram.axis.category["int"]([-1, 1], "charge");

axis_pdf_idx = ROOT.boost.histogram.axis.integer[""](0, 103, "pdf")

def build_graph(df, dataset):
    results = []

    if dataset.is_data:
        df = df.Define("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.DefinePerSample("one", "1.")
    hweight = df.Histo1D(("hweight", "", 1, 0.5, 1.5), "one", "weight")

    df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05")

    df = df.Filter("Sum(vetoMuons) == 1")

    df = df.Define("goodMuons", "vetoMuons && Muon_mediumId")

    df = df.Filter("Sum(goodMuons) == 1")

    df = df.Define("goodMuons_Pt0", "Muon_pt[goodMuons][0]")
    df = df.Define("goodMuons_Eta0", "Muon_eta[goodMuons][0]")
    df = df.Define("goodMuons_Charge0", "Muon_charge[goodMuons][0]")

    hptetacharge = df.HistoBoost("hptetacharge", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight"])
    results.append(hptetacharge)

    if not dataset.is_data:

        df = df.Define("pdfweight", "weight*LHEPdfWeight")

        df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")

        hptetachargepdf = df.HistoBoost("hptetachargepdf", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", "pdfweight"])
        results.append(hptetachargepdf)

    return results, hweight

narf.build_and_run(datasets, build_graph, "testout.root")
