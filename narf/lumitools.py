import csv
import json
import ROOT
import pathlib

ROOT.gInterpreter.Declare('#include "lumitools.h"')

def make_lumihelper(filename):
    runs = []
    lumis = []
    lumivals = []

    with open(filename) as lumicsv:
        reader = csv.reader(lumicsv)
        for row in reader:
            if row[0][0]=="#":
                continue

            run, _ = row[0].split(":")
            lumi, _ = row[1].split(":")
            lumival = row[6]
            
            run = int(run)
            lumi = int(lumi)
            lumival = float(lumival)
            
            runs.append(run)
            lumis.append(lumi)
            lumivals.append(lumival)            
        
    lumihelper = ROOT.LumiHelper(runs, lumis, lumivals)
    return lumihelper

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
