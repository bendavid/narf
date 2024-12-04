import numpy as np
import hist
import h5py
import tensorflow as tf
import narf.combineutils
import argparse
import narf.ioutils
import os

import pdb


parser =  argparse.ArgumentParser()

parser.add_argument("filename", help="filename of the main hdf5 input")
parser.add_argument("-o","--output", default="fitresults.hdf5",  help="output file name")
parser.add_argument("-t","--toys", default=-1, type=int, help="run a given number of toys, 0 fits the data, and -1 fits the asimov toy (the default)")
parser.add_argument("--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_argument("--POIMode", default="mu", help="mode for POI's")
parser.add_argument("--allowNegativePOI", default=False, action='store_true', help="allow signal strengths to be negative (otherwise constrained to be non-negative)")
parser.add_argument("--POIDefault", default=1., type=float, help="mode for POI's")
parser.add_argument("--saveHists", default=False, action='store_true', help="save prefit and postfit histograms")
parser.add_argument("--computeHistErrors", default=False, action='store_true', help="propagate uncertainties to prefit and postfit histograms")
parser.add_argument("--computeVariations", default=False, action='store_true', help="save postfit histograms with each noi varied up to down")
parser.add_argument("--binByBinStat", default=False, action='store_true', help="add bin-by-bin statistical uncertainties on templates (adding sumW2 on variance)")
parser.add_argument("--externalPostfit", default=None, help="load posfit nuisance parameters and covariance from result of an external fit")
parser.add_argument("--pseudoData", default=None, type=str, help="run fit on pseudo data with the given name")
parser.add_argument("--normalize", default=False, action='store_true', help="Normalize prediction and systematic uncertainties to the overall event yield in data")
parser.add_argument("--noChi2", default=False, action='store_true', help="Do not compute chi2 on prefit/postfit histograms")
parser.add_argument("--exp_transform", default=False, action='store_true', help="Reverse the exponential transform")
parser.add_argument("--project", nargs="+", action="append", default=[], help="add projection for the prefit and postfit histograms, specifying the channel name followed by the axis names, e.g. \"--project ch0 eta pt\".  This argument can be called multiple times")

args = parser.parse_args()
doChi2 = not args.noChi2

indata = narf.combineutils.FitInputData(args.filename, args.pseudoData)
fitter = narf.combineutils.Fitter(indata, args)

if args.toys == -1:
    fitter.nobs.assign(fitter.expected_events(profile=False))

cov_prefit = fitter.prefit_covariance()

results = {}

results["projections"] = []
for projection in args.project:
    channel = projection[0]
    axes = projection[1:]

    results["projections"].append({"channel" : channel, "axes" : axes})

if args.saveHists:

    hist_data_obs, hist_nobs = fitter.observed_hists()
    results.update({"hist_data_obs" : hist_data_obs,
                    "hist_nobs" : hist_nobs,})

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        hist_data_obs = results["hist_data_obs"][channel].get().project(*axes)
        hist_nobs = results["hist_nobs"][channel].get().project(*axes)

        hist_data_obs = narf.ioutils.H5PickleProxy(hist_data_obs)
        hist_nobs = narf.ioutils.H5PickleProxy(hist_nobs)

        projection.update({"hist_data_obs" : hist_data_obs,
                    "hist_nobs" : hist_nobs,})

    if doChi2:
        hist_prefit_inclusive, chi2_prefit, ndf_prefit = fitter.expected_hists(cov_prefit, inclusive=True, profile=False, compute_variance=args.computeHistErrors, compute_chi2=True, name = "prefit_inclusive", label = "prefit expected number of events for all processes combined")
    else:
        hist_prefit_inclusive = fitter.expected_hists(cov_prefit, inclusive=True, profile=False, compute_variance=args.computeHistErrors, name = "prefit_inclusive", label = "prefit expected number of events for all processes combined")
        chi2_prefit = 0.
        ndf_prefit = 1.

    hist_prefit = fitter.expected_hists(cov_prefit, inclusive=False, profile=False, compute_variance=args.computeHistErrors, name = "prefit", label = "prefit expected number of events")

    results.update({
        "hist_prefit_inclusive" : hist_prefit_inclusive,
        "hist_prefit" : hist_prefit,
        "ndf_prefit": ndf_prefit,
        "chi2_prefit": chi2_prefit,
    })

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        axes_str = "-".join(axes)

        if doChi2:
            hist_prefit_inclusive, chi2_prefit, ndf_prefit = fitter.expected_projection_hist(cov=cov_prefit, channel=channel, axes=axes, inclusive=True, profile=False, compute_variance=args.computeHistErrors, compute_chi2=True, name=f"prefit_inclusive_projection_{channel}_{axes_str}", label=f"prefit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)
        else:
            hist_prefit_inclusive = fitter.expected_projection_hist(cov=cov_prefit, channel=channel, axes=axes, inclusive=True, profile=False, compute_variance=args.computeHistErrors, name=f"prefit_inclusive_projection_{channel}_{axes_str}", label=f"prefit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)
            chi2_prefit = 0.
            ndf_prefit = 1.

        hist_prefit = fitter.expected_projection_hist(cov=cov_prefit, channel=channel, axes=axes, inclusive=False, profile=False, compute_variance=args.computeHistErrors, name=f"prefit_projection_{channel}_{axes_str}", label=f"prefit expected number of events, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)

        projection.update({
            "hist_prefit_inclusive" : hist_prefit_inclusive,
            "hist_prefit" : hist_prefit,
            "ndf_prefit": ndf_prefit,
            "chi2_prefit": chi2_prefit,
        })

    if args.computeVariations:
        cov_prefit_variations = fitter.prefit_covariance(unconstrained_err=1.)

        hist_prefit_variations = fitter.expected_hists(cov_prefit_variations, inclusive=True, profile=False, compute_variance=False, compute_variations=True, name = "prefit_inclusive_variations", label = "prefit expected number of events with variations of events for all processes combined")

        results["hist_prefit_variations"] = hist_prefit_variations

        for projection in results["projections"]:
            channel = projection["channel"]
            axes = projection["axes"]

            axes_str = "-".join(axes)

            hist_prefit_variations = fitter.expected_projection_hist(cov=cov_prefit_variations, channel=channel, axes=axes, inclusive=True, profile=False, compute_variance=False, compute_variations=True, name = f"prefit_inclusive_variations_projection_f{channel}_f{axes_str}", label = f"prefit expected number of events with variations of events for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)

            projection["hist_prefit_variations"] = hist_prefit_variations

        del cov_prefit_variations

dofit = args.toys >= 0 and args.externalPostfit is None

if dofit:
    fitter.minimize()

    val, grad, hess = fitter.loss_val_grad_hess()

    cov = tf.linalg.inv(hess)

if args.externalPostfit is not None:
    # load results from external fit and set postfit value and covariance elements for common parameters
    with h5py.File(args.externalPostfit, "r") as fext:
        parms_ext = fext["parms"][...]

        x_ext = fext["x"][...]
        cov_ext = fext["cov"][...]

    # FIXME do this without explicit loops and ideally in tensorflow directly

    xvals = fitter.x.numpy()
    covval = cov_prefit.numpy()

    parmmap = {}

    for iparm, parm in enumerate(fitter.parms):
        parmmap[parm] = iparm

    for iparm_ext, parmi_ext in enumerate(parms_ext):
        iparm = parmmap.get(parmi_ext)

        if iparm is None:
            continue

        xvals[iparm] = x_ext[iparm_ext]

        for jparm_ext, parmj_ext in enumerate(parms_ext):
            jparm = parmmap.get(parmj_ext)

            if jparm is None:
                continue

            covval[iparm, jparm] = cov_ext[iparm_ext, jparm_ext]

    fitter.x.assign(xvals)
    cov = tf.convert_to_tensor(covval)


postfit_profile = args.externalPostfit is None

nllvalfull = fitter.full_nll().numpy()
satnllvalfull, ndfsat = fitter.saturated_nll()

satnllvalfull = satnllvalfull.numpy()
ndfsat = ndfsat.numpy()

results.update({
    "nllvalfull" : nllvalfull,
    "satnllvalfull" : satnllvalfull,
    "ndfsat" : ndfsat,
    "postfit_profile" : postfit_profile,
})

if args.saveHists:

    if doChi2:
        hist_postfit_inclusive, chi2_postfit, ndf_postfit = fitter.expected_hists(cov, inclusive=True, profile=postfit_profile, compute_variance=args.computeHistErrors, compute_chi2=True, name = "postfit_inclusive", label = "postfit expected number of events for all processes combined")
    else:
        hist_postfit_inclusive = fitter.expected_hists(cov, inclusive=True, profile=postfit_profile, compute_variance=args.computeHistErrors, name = "postfit_inclusive", label = "postfit expected number of events for all processes combined")
        chi2_postfit = 0.
        ndf_postfit = 1.
        
    hist_postfit = fitter.expected_hists(cov, inclusive=False, profile=postfit_profile, compute_variance=args.computeHistErrors, name = "postfit", label = "postfit expected number of events")

    results.update({
        "hist_postfit_inclusive" : hist_postfit_inclusive,
        "hist_postfit" : hist_postfit,
        "ndf_postfit": ndf_postfit,
        "chi2_postfit": chi2_postfit,
    })

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        axes_str = "-".join(axes)
        
        if doChi2:
            hist_postfit_inclusive, chi2_postfit, ndf_postfit = fitter.expected_projection_hist(cov=cov, channel=channel, axes=axes, inclusive=True, profile=postfit_profile, compute_variance=args.computeHistErrors, compute_chi2=doChi2, name=f"postfit_inclusive_projection_{channel}_{axes_str}", label=f"postfit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)
        else:
            hist_postfit_inclusive = fitter.expected_projection_hist(cov=cov, channel=channel, axes=axes, inclusive=True, profile=postfit_profile, compute_variance=args.computeHistErrors, name=f"postfit_inclusive_projection_{channel}_{axes_str}", label=f"postfit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)
            chi2_postfit = 0.
            ndf_postfit = 1.
        
        hist_postfit = fitter.expected_projection_hist(cov=cov, channel=channel, axes=axes, inclusive=False, profile=postfit_profile, compute_variance=args.computeHistErrors, name=f"postfit_projection_{channel}_{axes_str}", label=f"postfit expected number of events, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)

        projection.update({
            "hist_postfit_inclusive" : hist_postfit_inclusive,
            "hist_postfit" : hist_postfit,
            "ndf_postfit": ndf_postfit,
            "chi2_postfit": chi2_postfit,
        })

    if args.computeVariations:
        hist_postfit_variations = fitter.expected_hists(cov, inclusive=True, profile=postfit_profile, profile_grad=False, compute_variance=False, compute_variations=True, name = "postfit_inclusive_variations", label = "postfit expected number of events with variations of events for all processes combined")

        results["hist_postfit_variations"] = hist_postfit_variations

        hist_postfit_variations_correlated = fitter.expected_hists(cov, inclusive=True, profile=postfit_profile, compute_variance=False, compute_variations=True, correlated_variations=True, name = "hist_postfit_variations_correlated", label = "postfit expected number of events with variations of events (including correlations) for all processes combined")

        results["hist_postfit_variations_correlated"] = hist_postfit_variations_correlated

        for projection in results["projections"]:
            channel = projection["channel"]
            axes = projection["axes"]

            axes_str = "-".join(axes)

            hist_postfit_variations = fitter.expected_projection_hist(cov=cov, channel=channel, axes=axes, inclusive=True, profile=postfit_profile, profile_grad=False, compute_variance=False, compute_variations=True, name = f"postfit_inclusive_variations_projection_f{channel}_f{axes_str}", label = f"postfit expected number of events with variations of events for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)

            projection["hist_postfit_variations"] = hist_postfit_variations

            hist_postfit_variations_correlated = fitter.expected_projection_hist(cov=cov, channel=channel, axes=axes, inclusive=True, profile=postfit_profile, compute_variance=False, compute_variations=True, correlated_variations=True, name = f"postfit_inclusive_variations_correlated_projection_f{channel}_f{axes_str}", label = f"postfit expected number of events with variations of events (including correlations) for all processes combined, projection for channel {channel} and axes {axes_str}.", exp_transform=args.exp_transform)

            projection["hist_postfit_variations_correlated"] = hist_postfit_variations_correlated

# pass meta data into output file
meta = {
    "meta_info" : narf.ioutils.make_meta_info_dict(args=args), 
    "meta_info_input": fitter.indata.metadata,
    "signals": fitter.indata.signals,
    "procs": fitter.indata.procs,
}

outfolder = os.path.dirname(args.output)
if outfolder:
    if not os.path.exists(outfolder):
        print(f"Creating output folder {outfolder}")
        os.makedirs(outfolder)
    
with h5py.File(args.output, "w") as fout:
    narf.ioutils.pickle_dump_h5py("results", results, fout)
    narf.ioutils.pickle_dump_h5py("meta", meta, fout)
