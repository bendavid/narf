import csv
import json
from datetime import datetime
import ROOT
import pathlib
import narf.clingutils

narf.clingutils.Declare('#include "lumitools.hpp"')


def make_brilcalc_helper(filename, value, action):
    runs = []
    lumis = []
    vals = []

    with open(filename) as lumicsv:
        _ = next(lumicsv) # skip first header, e.g., '#Data tag : 24v2 , Norm tag: None\n'
        reader = csv.DictReader(lumicsv)
        for row in reader:
            if row['#run:fill'][0]=="#":
                continue

            run, _ = row['#run:fill'].split(":")
            lumi, _ = row['ls'].split(":")
            val = row[value]
            
            run = int(run)
            lumi = int(lumi)
            val = action(val)
            
            runs.append(run)
            lumis.append(lumi)
            vals.append(val)  
        
    brilcalc_helper = ROOT.BrilcalcHelper(runs, lumis, vals)
    return brilcalc_helper


def make_lumihelper(filename):
    return make_brilcalc_helper(filename, value='recorded(/fb)', action=float)


def make_dedup_lumihelper(filename):
    """Like make_lumihelper but returns 0 for repeated (run, ls) pairs.

    Use this when the LuminosityBlocks TChain contains duplicate entries
    (e.g. low-PU NanoAOD where CRAB merges several MINIAOD input files per
    output file, recording all their luminosity blocks even when they contain
    no selected events).  The underlying C++ class uses a shared mutex so
    deduplication is correct under ROOT's implicit multi-threading.
    """
    runs = []
    lumis = []
    vals = []

    with open(filename) as lumicsv:
        _ = next(lumicsv)
        reader = csv.DictReader(lumicsv)
        for row in reader:
            if row['#run:fill'][0] == "#":
                continue
            run, _ = row['#run:fill'].split(":")
            lumi, _ = row['ls'].split(":")
            runs.append(int(run))
            lumis.append(int(lumi))
            vals.append(float(row['recorded(/fb)']))

    return ROOT.DeduplicatingBrilcalcHelper(runs, lumis, vals)


def make_timehelper(filename):
    action = lambda x: datetime.strptime(x, "%m/%d/%y %H:%M:%S").timestamp()
    return make_brilcalc_helper(filename, value='time', action=action)


def make_jsonhelper(filename):

    with open(filename) as jsonfile:
        jsondata = json.load(jsonfile)
    
    runs = []
    firstlumis = []
    lastlumis = []
    
    for run,lumipairs in jsondata.items():
        for lumipair in lumipairs:
            runs.append(int(run))
            firstlumis.append(int(lumipair[0]))
            lastlumis.append(int(lumipair[1]))
    
    jsonhelper = ROOT.JsonHelper(runs, firstlumis, lastlumis)
    
    return jsonhelper
