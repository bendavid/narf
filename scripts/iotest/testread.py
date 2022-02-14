import hist
import lz4.frame
import pickle

fname = "mw_with_mu_eta_pt.pkl.lz4"

with lz4.frame.open(fname, "rb") as f:
    results = pickle.load(f)


