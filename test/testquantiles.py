import numpy as np
import scipy
import hist
import narf
import ROOT
import awkward as ak

ncond = 2
nquant = 2

nsamples = int(10e6)
nbins = 10


rng = np.random.default_rng(12345)

ndim = ncond+nquant

# generate data sample which is uniform in the conditional variables and an nd camel function in the remaining
# variables to be transformed

data_cond = rng.uniform(size=(nsamples, ndim))

data_switch = rng.integers(low=0, high=2, size=(nsamples))
loc = np.where(data_switch==0., 1./3., 2./3.)
loc = loc[:, None]

data_quant = rng.normal(loc=loc, scale=0.1/np.sqrt(2.), size=(nsamples, nquant))

print(data_cond)
print(data_quant)

# make awkward array with named records

data_split = []
for i in range(ncond):
    data_split.append(data_cond[:, i])
for i in range(nquant):
    data_split.append(data_quant[:, i])

akdict = {}
cols = []
for i, idata in enumerate(data_split):
    akdict[f"col_{i}"] = idata

# convert awkward array to rdf
df = ak.to_rdataframe(akdict)

ROOT.ROOT.RDF.Experimental.AddProgressBar(df)

cols = list(akdict.keys())

condcols = cols[:ncond]
quantcols = cols[ncond:]

axes = [hist.axis.Regular(nbins, 0., 1., name=f"variable_{i}") for i in range(ndim)]

condaxes = axes[:ncond]
quantaxes = [hist.axis.Regular(nbins, 0., 1., name=f"quantile_{i}", underflow=False, overflow=False) for i in range(nquant)]

# build quantile hists (triggers an event loop)
quantile_hists = narf.histutils.build_quantile_hists(df, cols, condaxes, quantaxes)

# define transformed variables with quantile bin indexes
df, quantile_axes, quantile_cols = narf.histutils.define_quantile_ints(df, cols=condcols+quantcols, quantile_hists=quantile_hists)

# book histograms
horig = df.HistoBoost("horig", axes, cols)
hquantiles = df.HistoBoost("hquantiles", quantile_axes, quantile_cols)

# trigger the event loop
horig = horig.GetValue()
hquantiles = hquantiles.GetValue()

print(horig)
print(hquantiles)

def stats(h):
    vals = np.ravel(h.values())
    mean = np.mean(vals)
    std = np.std(vals)

    print(h.name, "total:", np.sum(vals), "mean:", mean, "std:", std)

stats(horig)
stats(hquantiles)
