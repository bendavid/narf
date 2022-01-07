import ROOT
import boost_histogram as bh
import hist
from hashlib import sha256
import time
import math
import numpy as np
import functools
#import cppyy.ll

ROOT.gInterpreter.Declare('#include "histutils.h"')
ROOT.gInterpreter.Declare('#include "FillBoostHelperAtomic.h"')

def bool_to_string(b):
    if b:
        return "true"
    else:
        return "false"

def convert_axis(axis):
    default = ROOT.boost.histogram.use_default

    if (axis.traits.growth):
        raise ValueError("growable axes are not supported")

    optionsval = ROOT.narf.get_option[bool_to_string(axis.traits.underflow),
                                   bool_to_string(axis.traits.overflow),
                                   bool_to_string(axis.traits.circular),
                                   bool_to_string(axis.traits.growth)]()

    options = type(optionsval)

    if isinstance(axis, bh.axis.Regular):
        #TODO add transform support
        if axis.transform is not None:
            raise ValueError("transforms are not currently supported")

        nbins = axis.size
        xlow = axis.edges[0]
        xhigh = axis.edges[-1]

        return ROOT.boost.histogram.axis.regular["double", default, default, options](nbins, xlow, xhigh)
    elif isinstance(axis, bh.axis.Variable):
        return ROOT.boost.histogram.axis.variable["double", default, options](axis.edges)
    elif isinstance(axis, bh.axis.Integer):
        ilow = axis.bin(0)
        ihigh = axis.bin(axis.size - 1) + 1
        return ROOT.boost.histogram.axis.integer["int", default, options](ilow, ihigh)
    elif isinstance(axis, bh.axis.IntCategory):
        ncats = axis.size
        cats = [axis.bin(icat) for icat in range(ncats)]
        return ROOT.boost.histogram.axis.category["int", default, options](cats)
    elif isinstance(axis, bh.axis.StrCategory):
        ncats = axis.size
        cats = [axis.bin(icat) for icat in range(ncats)]
        return ROOT.boost.histogram.axis.category[ROOT.std.string, default, options](cats)
    elif isinstance(axis, bh.axis.Boolean):
        return ROOT.boost.histogram.axis.boolean[""]()
    else:
        raise TypeError("axis must be a boost_histogram or compatible axis")

def convert_storage_type(storage, force_atomic):
    if isinstance(storage, bh.storage.Double):
        if force_atomic:
            return "narf::atomic_adaptor<double>"
        else:
            return "double"
    elif isinstance(storage, bh.storage.Unlimited):
        raise TypeError("Unlimited storage not supported")
    elif isinstance(storage, bh.storage.Int64):
        if force_atomic:
            return "narf::atomic_adaptor<std::int64_t>"
        else:
            return "std::int64_t"
    elif isinstance(storage, bh.storage.AtomicInt64):
        return "boost::histogram::accumulators::count<std::int64_t, true>"
    elif isinstance(storage, bh.storage.Weight):
        if force_atomic:
            return "boost::histogram::accumulators::weighted_sum<narf::atomic_adaptor<double>>"
        else:
            return "boost::histogram::accumulators::weighted_sum<double>"
    elif isinstance(storage, bh.storage.Mean):
        if force_atomic:
            return "narf::atomic_adaptor<boost::histogram::accumulators::mean<double>>"
        else:
            return "boost::histogram::accumulators::mean<double>"
    elif isinstance(storage, bh.storage.WeightedMean):
        if force_atomic:
            return "narf::atomic_adaptor<boost::histogram::accumulators::weighted_mean<double>>"
        else:
            return "boost::histogram::accumulators::weighted_mean<double>"
    else:
        raise TypeError("storage must be a boost_histogram or compatible storage type")

def _histo_boost(df, name, axes, cols, storage = bh.storage.Weight(), force_atomic = ROOT.ROOT.IsImplicitMTEnabled()):
    # first construct a histogram from the hist python interface, then construct a boost histogram
    # using PyROOT with compatible axes and storage types, adopting the underlying storage
    # of the python hist histogram

    #storage = bh.storage.Mean()

    _hist = hist.Hist(*axes, storage = storage)

    arr = _hist.view(flow = True).__array_interface__

    addr = arr["data"][0]
    shape = arr["shape"]
    elem_size = int(arr["typestr"][2:])
    strides = arr["strides"]

    size = math.prod(shape)
    size_bytes = size*elem_size

    # compute strides for a fortran-style contiguous array with the given shape
    stridesf = []
    current_stride = elem_size
    for axis_size in shape:
        stridesf.append(current_stride)
        current_stride *= axis_size
    stridesf = tuple(stridesf)

    if strides is None:
        #default stride for C-style contiguous array
        strides = tuple(reversed(stridesf))

    # check that memory is a fortran-style contiguous array
    if strides != stridesf:
        raise ValueError("memory is not a contiguous fortran-style array as required by the C++ class")

    #cppaxes = [convert_axis(axis) for axis in axes]
    cppaxes = [ROOT.std.move(convert_axis(axis)) for axis in axes]
    cppstoragetype = convert_storage_type(storage, force_atomic)

    h = ROOT.narf.make_histogram_adopted[cppstoragetype](addr, size_bytes, *cppaxes)

    #bufptr = cppyy.ll.reinterpret_cast["void*"](addr)
    #storage = ROOT.narf.adopted_storage[cppstoragetype](bufptr, size_bytes)
    #h = ROOT.narf.make_histogram_with_storage(ROOT.std.move(storage), *cppaxes)

    # confirm storage order empirically
    origin = (0,)*len(shape)
    origin_addr = ROOT.addressof(h.at(*origin))

    for iaxis, stride in enumerate(strides):
        coords = [0,]*len(shape)
        coords[iaxis] = 1
        addr = ROOT.addressof(h.at(*coords))
        addr_diff = addr - origin_addr
        if addr_diff != stride:
            raise ValueError("mismatched storage ordering")

    helper = ROOT.narf.FillBoostHelperAtomic[type(h)](ROOT.std.move(h))
    coltypes = [df.GetColumnType(col) for col in cols]
    targs = tuple([type(df), type(helper)] + coltypes)
    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)

    res.name = name
    res._hist = _hist

    # hide underlying C++ class and return the python version instead

    res._GetValue = res.GetValue

    def get_hist():
        res._GetValue()
        return res._hist

    def hist_getitem(*args, **kwargs):
        res._GetValue()
        return res._hist.__getitem__(*args, **kwargs)

    ret_null = lambda : None

    res.__deref__ = get_hist
    res.__follow__ = get_hist
    res.begin = ret_null
    res.end = ret_null
    res.GetPtr = get_hist
    res.GetValue = get_hist
    res.__getitem__ = hist_getitem

    return res


def _convert_root_axis(axis):
    is_regular = axis.GetXbins().fN == 0

    if is_regular:
        nbins = axis.GetNbins()
        xlow = axis.GetXmin()
        xhigh = axis.GetXmax()
        return ROOT.boost.histogram.axis.regular[""](nbins, xlow, xhigh)
    else:
        edges = [edge for edge in axis.GetXbins()]
        return ROOT.boost.histogram.axis.variable[""](edges)

def _convert_root_hist(hist):
    axes = []
    if isinstance(hist, ROOT.TH3):
        axes.append(hist.GetXaxis())
        axes.append(hist.GetYaxis())
        axes.append(hist.GetZaxis())
    elif isinstance(hist, ROOT.TH2):
        axes.append(hist.GetXaxis())
        axes.append(hist.GetYaxis())
    elif isinstance(hist, ROOT.TH1):
        axes.append(hist.GetXaxis())
    elif isinstance(hist, ROOT.THnBase):
        for axis in hist.GetListOfAxes():
            axes.append(axis)

    boost_axes = [_convert_root_axis(axis) for axis in axes]
    boost_hist = ROOT.narf.make_atomic_histogram_with_error(*boost_axes)

    return boost_hist

def _convert_root_axis_info(nbins, xlow, xhigh, edges):

    if edges:
        return ROOT.boost.histogram.axis.variable[""](edges)
    else:
        return ROOT.boost.histogram.axis.regular[""](nbins, xlow, xhigh)


def _convert_root_model(model):

    axes_info = []

    if isinstance(model, ROOT.ROOT.RDF.TH1DModel):
        hist_type = ROOT.TH1D
        axes_info.append((model.fNbinsX, model.fXLow, model.fXUp, model.fBinXEdges))
    elif isinstance(model, ROOT.ROOT.RDF.TH2DModel):
        hist_type = ROOT.TH2D
        axes_info.append((model.fNbinsX, model.fXLow, model.fXUp, model.fBinXEdges))
        axes_info.append((model.fNbinsY, model.fYLow, model.fYUp, model.fBinYEdges))
    elif isinstance(model, ROOT.ROOT.RDF.TH3DModel):
        hist_type = ROOT.TH3D
        axes_info.append((model.fNbinsX, model.fXLow, model.fXUp, model.fBinXEdges))
        axes_info.append((model.fNbinsY, model.fYLow, model.fYUp, model.fBinYEdges))
        axes_info.append((model.fNbinsZ, model.fZLow, model.fZUp, model.fBinZEdges))
    elif isinstance(model, ROOT.ROOT.RDF.THnDModel):
        hist_type = ROOT.THnT["double"]
        for nbins, xlow, xhigh, edges in zip(model.fNbins, model.fXmin, model.fXmax, model.fBinEdges):
            axes_info.append((nbins, xlow, xhigh, edges))

    boost_axes = [_convert_root_axis_info(*axis_info) for axis_info in axes_info]
    boost_hist = ROOT.narf.make_atomic_histogram_with_error(*boost_axes)

    return hist_type, boost_hist



def _histo_with_boost(df, model, cols):

    hist_type, boost_hist = _convert_root_model(model)
    helper = ROOT.narf.FillBoostHelperAtomic[hist_type, type(boost_hist)](model, ROOT.std.move(boost_hist))

    coltypes = [df.GetColumnType(col) for col in cols]
    targs = tuple([type(df), type(helper)] + coltypes)
    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)

    return res

def _histo1d_with_boost(df, model, v, w = None):
    if isinstance(model, tuple) or isinstance(model, list):
        model = ROOT.RDF.TH1DModel(*model)

    cols = [v]
    if w is not None:
        cols.append(w)

    return _histo_with_boost(df, model, cols)

def _histo2d_with_boost(df, model, v0, v1,  w = None):
    if isinstance(model, tuple) or isinstance(model, list):
        model = ROOT.RDF.TH2DModel(*model)

    cols = [v0, v1]
    if w is not None:
        cols.append(w)

    return _histo_with_boost(df, model, cols)

def _histo3d_with_boost(df, model, v0, v1, v2, w = None):
    if isinstance(model, tuple) or isinstance(model, list):
        model = ROOT.RDF.TH3DModel(*model)

    cols = [v0, v1, v2]
    if w is not None:
        cols.append(w)

    return _histo_with_boost(df, model, cols)

def _histond_with_boost(df, model, cols):
    if isinstance(model, tuple) or isinstance(model, list):
        model = ROOT.RDF.THnDModel(*model)

    return _histo_with_boost(df, model, cols)

def _sum_and_count(df, col):
    sumres = df.Sum(col)
    countres = df.Count()
    return (sumres, countres)

@ROOT.pythonization("RInterface<", ns="ROOT::RDF", is_prefix=True)
def pythonize_rdataframe(klass):
    # add function for boost histograms
    klass.HistoBoost = _histo_boost
    klass.Histo1DWithBoost = _histo1d_with_boost
    klass.Histo2DWithBoost = _histo2d_with_boost
    klass.Histo3DWithBoost = _histo3d_with_boost
    klass.HistoNDWithBoost = _histond_with_boost
    klass.SumAndCount = _sum_and_count
