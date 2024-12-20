import numpy as np
import hist
import h5py
import tensorflow as tf
import narf.combineutils
import argparse
import narf.ioutils

import pdb


parser =  argparse.ArgumentParser()

parser.add_argument("filename", help="filename of the main hdf5 input")
parser.add_argument("-o","--output", default="fitresults",  help="output file name")
parser.add_argument("--outputFormat", default="narf", choices=["narf", "h5py"],  help="output file name")
parser.add_argument("-t","--toys", default=-1, type=int, help="run a given number of toys, 0 fits the data, and -1 fits the asimov toy (the default)")
parser.add_argument("--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_argument("--POIMode", default="mu", help="mode for POI's")
parser.add_argument("--allowNegativePOI", default=False, action='store_true', help="allow signal strengths to be negative (otherwise constrained to be non-negative)")
parser.add_argument("--POIDefault", default=1., type=float, help="mode for POI's")
parser.add_argument("--saveHists", default=False, action='store_true', help="save prefit and postfit histograms")
parser.add_argument("--computeHistErrors", default=False, action='store_true', help="propagate uncertainties to prefit and postfit histograms")
parser.add_argument("--computeHistCov", default=False, action='store_true', help="propagate covariance of histogram bins (inclusive in processes)")
parser.add_argument("--computeHistImpacts", default=False, action='store_true', help="propagate global impacts on histogram bins (inclusive in processes)")
parser.add_argument("--computeVariations", default=False, action='store_true', help="save postfit histograms with each noi varied up to down")
parser.add_argument("--noChi2", default=False, action='store_true', help="Do not compute chi2 on prefit/postfit histograms")
parser.add_argument("--binByBinStat", default=False, action='store_true', help="add bin-by-bin statistical uncertainties on templates (adding sumW2 on variance)")
parser.add_argument("--externalPostfit", default=None, help="load posfit nuisance parameters and covariance from result of an external fit")
parser.add_argument("--pseudoData", default=None, type=str, help="run fit on pseudo data with the given name")
parser.add_argument("--normalize", default=False, action='store_true', help="Normalize prediction and systematic uncertainties to the overall event yield in data")
parser.add_argument("--project", nargs="+", action="append", default=[], help="add projection for the prefit and postfit histograms, specifying the channel name followed by the axis names, e.g. \"--project ch0 eta pt\".  This argument can be called multiple times")
parser.add_argument("--doImpacts", default=False, action='store_true', help="Compute impacts on POIs per nuisance parameter and per-nuisance parameter group")
parser.add_argument("--globalImpacts", default = False, action='store_true', help="compute impacts in terms of variations of global observables (as opposed to nuisance parameters directly)")
parser.add_argument("--chisqFit", default=False, action='store_true',  help="Perform chi-square fit instead of likelihood fit")
parser.add_argument("--externalCovariance", default=False, action='store_true',  help="Using an external covariance matrix for the observations in the chi-square fit")

args = parser.parse_args()

indata = narf.combineutils.FitInputData(args.filename, args.pseudoData)
workspace = narf.combineutils.Workspace(args.outputFormat)
fitter = narf.combineutils.Fitter(indata, args, workspace)

if args.toys == -1:
    fitter.nobs.assign(fitter.expected_events(profile=False))

cov_prefit = fitter.prefit_covariance()

results = {
    "parms_prefit": fitter.parms_hist(cov_prefit, hist_name="prefit"),
    "projections": [{"channel": projection[0], "axes": projection[1:]} for projection in args.project],
}

if args.saveHists:
    print("Save prefit hists")

    hist_data_obs, hist_nobs = fitter.observed_hists()
    results.update({"hist_data_obs" : hist_data_obs,
                    "hist_nobs" : hist_nobs,})

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        hist_data_obs = fitter.workspace.project(results["hist_data_obs"][channel], axes)
        hist_nobs = fitter.workspace.project(results["hist_nobs"][channel], axes)

        projection.update({"hist_data_obs" : hist_data_obs,
                    "hist_nobs" : hist_nobs,})

    print(f"Save - inclusive hist")

    hist_prefit_inclusive, aux_info = fitter.expected_hists(
        cov_prefit, 
        inclusive=True, 
        profile=False, 
        compute_variance=args.computeHistErrors, 
        compute_chi2=not args.noChi2, 
        aux_info=True,
        name = "prefit_inclusive", 
        label = "prefit expected number of events for all processes combined",
        )

    print(f"Save - processes hist")

    hist_prefit = fitter.expected_hists(
        cov_prefit, 
        inclusive=False, 
        profile=False, 
        compute_variance=args.computeHistErrors, 
        name = "prefit", 
        label = "prefit expected number of events",
        )

    results.update({
        "hist_prefit_inclusive" : hist_prefit_inclusive,
        "hist_prefit" : hist_prefit,

    })
    if not args.noChi2:
        results["ndf_prefit"] = aux_info["ndf"]
        results["chi2_prefit"] = aux_info["chi2"]

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        print(f"Save projection for channel {channel} - inclusive")

        axes_str = "-".join(axes)

        hist_prefit_inclusive, aux_info = fitter.expected_projection_hist(
            cov=cov_prefit, 
            channel=channel, 
            axes=axes, 
            inclusive=True, 
            profile=False, 
            compute_variance=args.computeHistErrors, 
            compute_chi2=not args.noChi2, 
            aux_info=True,
            name=f"prefit_inclusive_projection_{channel}_{axes_str}", 
            label=f"prefit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
            )

        print(f"Save projection for channel {channel} - processes")

        hist_prefit = fitter.expected_projection_hist(
            cov=cov_prefit, 
            channel=channel, 
            axes=axes, 
            inclusive=False, 
            profile=False, 
            compute_variance=args.computeHistErrors,
            name=f"prefit_projection_{channel}_{axes_str}", label=f"prefit expected number of events, projection for channel {channel} and axes {axes_str}.",
            )

        projection.update({
            "hist_prefit_inclusive" : hist_prefit_inclusive,
            "hist_prefit" : hist_prefit
        })
        if not args.noChi2:
            projection["ndf_prefit"] = aux_info["ndf"]
            projection["chi2_prefit"] = aux_info["chi2"]

    if args.computeVariations:
        cov_prefit_variations = fitter.prefit_covariance(unconstrained_err=1.)

        hist_prefit_variations = fitter.expected_hists(cov_prefit_variations, inclusive=True, profile=False, compute_variance=False, compute_variations=True, name = "prefit_inclusive_variations", label = "prefit expected number of events with variations of events for all processes combined")

        results["hist_prefit_variations"] = hist_prefit_variations

        for projection in results["projections"]:
            channel = projection["channel"]
            axes = projection["axes"]

            axes_str = "-".join(axes)

            hist_prefit_variations = fitter.expected_projection_hist(cov=cov_prefit_variations, channel=channel, axes=axes, inclusive=True, profile=False, compute_variance=False, compute_variations=True, name = f"prefit_inclusive_variations_projection_f{channel}_f{axes_str}", label = f"prefit expected number of events with variations of events for all processes combined, projection for channel {channel} and axes {axes_str}.")

            projection["hist_prefit_variations"] = hist_prefit_variations

        del cov_prefit_variations


if args.externalPostfit is not None:
    # load results from external fit and set postfit value and covariance elements for common parameters
    with h5py.File(args.externalPostfit, "r") as fext:
        if "x" in fext.keys():
            # fitresult from combinetf 1
            x_ext = fext["x"][...]
            parms_ext = fext["parms"][...].astype(str)
            cov_ext = fext["cov"][...]
        else:
            # fitresult from combinetf 2
            h5results_ext = narf.ioutils.pickle_load_h5py(fext["results"])
            h_parms_ext = h5results_ext["parms"].get()

            x_ext = h_parms_ext.values()
            parms_ext = np.array(h_parms_ext.axes["parms"])
            cov_ext = h5results_ext["cov"].get().values()

    # FIXME do this without explicit loops and ideally in tensorflow directly

    xvals = fitter.x.numpy()
    covval = cov_prefit.numpy()

    parmmap = {}

    parms = fitter.parms.astype(str)
    for iparm, parm in enumerate(parms):
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
else:
    dofit = args.toys >= 0

    if dofit:
        fitter.minimize()

    val, grad, hess = fitter.loss_val_grad_hess()
    cov = tf.linalg.inv(hess)

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
    "cov": fitter.cov_hist(cov),
    "parms": fitter.parms_hist(cov),
})

if args.doImpacts:

    h, h_grouped = fitter.impacts_hists(cov, hess)
    results["impacts"] = h
    results["impacts_grouped"] = h_grouped

    if args.globalImpacts:

        h, h_grouped = fitter.global_impacts_hists(cov)
        results["global_impacts"] = h
        results["global_impacts_grouped"] = h_grouped

if args.saveHists:
    print("Save postfit hists")

    print(f"Save - inclusive hist")

    hist_postfit_inclusive, aux_info = fitter.expected_hists(
        cov, 
        inclusive=True, 
        profile=postfit_profile, 
        compute_variance=args.computeHistErrors, 
        compute_cov=args.computeHistCov,
        compute_global_impacts=args.computeHistImpacts,
        compute_chi2=not args.noChi2,
        aux_info=True,
        name = "postfit_inclusive", 
        label = "postfit expected number of events for all processes combined",
        )

    print(f"Save - processes hist")

    hist_postfit = fitter.expected_hists(
        cov, 
        inclusive=False, 
        profile=postfit_profile, 
        compute_variance=args.computeHistErrors, 
        name = "postfit", 
        label = "postfit expected number of events",
        )

    results.update({
        "hist_postfit_inclusive" : hist_postfit_inclusive,
        "hist_postfit" : hist_postfit,
    })
    if not args.noChi2:
        results["ndf_postfit"] = aux_info["ndf"]
        results["chi2_postfit"] = aux_info["chi2"]
    if args.computeHistCov:
        results["hist_cov_postfit_inclusive"] = aux_info["hist_cov"]    
    if args.computeHistImpacts:
        results["hist_global_impacts_postfit_inclusive"] = aux_info["hist_global_impacts"]
        results["hist_global_impacts_grouped_postfit_inclusive"] = aux_info["hist_global_impacts_grouped"]

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        axes_str = "-".join(axes)

        print(f"Save projection for channel {channel} - inclusive")

        hist_postfit_inclusive, aux_info = fitter.expected_projection_hist(
            cov=cov, 
            channel=channel, 
            axes=axes, 
            inclusive=True, 
            profile=postfit_profile, 
            compute_variance=args.computeHistErrors, 
            compute_cov=args.computeHistCov,
            compute_global_impacts=args.computeHistImpacts,
            compute_chi2=not args.noChi2,
            aux_info=True,
            name=f"postfit_inclusive_projection_{channel}_{axes_str}", 
            label=f"postfit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
            )

        print(f"Save projection for channel {channel} - inclusive")

        hist_postfit = fitter.expected_projection_hist(
            cov=cov, 
            channel=channel, 
            axes=axes, 
            inclusive=False, 
            profile=postfit_profile, 
            compute_variance=args.computeHistErrors, 
            name=f"postfit_projection_{channel}_{axes_str}", 
            label=f"postfit expected number of events, projection for channel {channel} and axes {axes_str}.",
            )

        projection.update({
            "hist_postfit_inclusive" : hist_postfit_inclusive,
            "hist_postfit" : hist_postfit,
        })
        if not args.noChi2:
            projection["ndf_postfit"] = aux_info["ndf"]
            projection["chi2_postfit"] = aux_info["chi2"]
        if args.computeHistCov:
            projection["hist_cov_postfit_inclusive"] = aux_info["hist_cov"]
        if args.computeHistImpacts:
            projection["hist_global_impacts_postfit_inclusive"] = aux_info["hist_global_impacts"]
            projection["hist_global_impacts_grouped_postfit_inclusive"] = aux_info["hist_global_impacts_grouped"]

    if args.computeVariations:
        hist_postfit_variations = fitter.expected_hists(
            cov, 
            inclusive=True, 
            profile=postfit_profile, 
            profile_grad=False, 
            compute_variance=False, 
            compute_variations=True, 
            name = "postfit_inclusive_variations", 
            label = "postfit expected number of events with variations of events for all processes combined",
            )

        results["hist_postfit_variations"] = hist_postfit_variations

        hist_postfit_variations_correlated = fitter.expected_hists(
            cov, 
            inclusive=True, 
            profile=postfit_profile, 
            compute_variance=False, 
            compute_variations=True, 
            correlated_variations=True, 
            name = "hist_postfit_variations_correlated", 
            label = "postfit expected number of events with variations of events (including correlations) for all processes combined",
            )

        results["hist_postfit_variations_correlated"] = hist_postfit_variations_correlated

        for projection in results["projections"]:
            channel = projection["channel"]
            axes = projection["axes"]

            axes_str = "-".join(axes)

            hist_postfit_variations = fitter.expected_projection_hist(
                cov=cov, 
                channel=channel, 
                axes=axes, 
                inclusive=True, 
                profile=postfit_profile, 
                profile_grad=False, 
                compute_variance=False, 
                compute_variations=True, 
                name = f"postfit_inclusive_variations_projection_f{channel}_f{axes_str}", 
                label = f"postfit expected number of events with variations of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
                )

            projection["hist_postfit_variations"] = hist_postfit_variations

            hist_postfit_variations_correlated = fitter.expected_projection_hist(
                cov=cov, 
                channel=channel, 
                axes=axes, 
                inclusive=True, 
                profile=postfit_profile, 
                compute_variance=False, 
                compute_variations=True, 
                correlated_variations=True, 
                name = f"postfit_inclusive_variations_correlated_projection_f{channel}_f{axes_str}", 
                label = f"postfit expected number of events with variations of events (including correlations) for all processes combined, projection for channel {channel} and axes {axes_str}.",
                )

            projection["hist_postfit_variations_correlated"] = hist_postfit_variations_correlated


# pass meta data into output file
meta = {
    "meta_info" : narf.ioutils.make_meta_info_dict(args=args), 
    "meta_info_input": fitter.indata.metadata,
    "signals": fitter.indata.signals,
    "procs": fitter.indata.procs,
}

fitter.workspace.write(args.output, results, meta)
