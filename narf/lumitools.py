import csv
import json
from datetime import datetime
import ROOT
import pathlib
import narf.clingutils

narf.clingutils.Declare('#include "lumitools.hpp"')


def make_brilcalc_helper(filename, idx, action):
    runs = []
    lumis = []
    vals = []

    with open(filename) as lumicsv:
        reader = csv.reader(lumicsv)
        for row in reader:
            if row[0][0]=="#":
                continue

            run, _ = row[0].split(":")
            lumi, _ = row[1].split(":")
            val = row[idx]
            
            run = int(run)
            lumi = int(lumi)
            val = action(val)
            
            runs.append(run)
            lumis.append(lumi)
            vals.append(val)            
        
    brilcalc_helper = ROOT.BrilcalcHelper(runs, lumis, vals)
    return brilcalc_helper


def make_lumihelper(filename):
    return make_brilcalc_helper(filename, idx=6, action=float)


def make_timehelper(filename):
    action = lambda x: datetime.strptime(x, "%m/%d/%y %H:%M:%S").timestamp()
    return make_brilcalc_helper(filename, idx=2, action=action)


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
