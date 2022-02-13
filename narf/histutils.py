import ROOT
import boost_histogram as bh
import hist
from hashlib import sha256
import time
import math
import numpy as np
import functools
import uuid
import array
import cppyy.ll

ROOT.gInterpreter.Declare('#include "histutils.h"')
ROOT.gInterpreter.Declare('#include "FillBoostHelperAtomic.h"')

ROOT.gInterpreter.Declare('#include <eigen3/Eigen/Dense>')
ROOT.gInterpreter.Declare('#include <eigen3/unsupported/Eigen/CXX11/Tensor>')

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
        return ROOT.narf.make_variable_axis["double", default, options](axis.edges)
    elif isinstance(axis, bh.axis.Integer):
        ilow = axis.bin(0)
        ihigh = axis.bin(axis.size - 1) + 1
        return ROOT.boost.histogram.axis.integer["int", default, options](ilow, ihigh)
    elif isinstance(axis, bh.axis.IntCategory):
        ncats = axis.size
        cats = [axis.bin(icat) for icat in range(ncats)]
        return ROOT.narf.make_category_axis["int", default, options](cats)
    elif isinstance(axis, bh.axis.StrCategory):
        ncats = axis.size
        cats = [axis.bin(icat) for icat in range(ncats)]
        return ROOT.narf.make_category_axis[ROOT.std.string, default, options](cats)
    elif isinstance(axis, bh.axis.Boolean):
        return ROOT.boost.histogram.axis.boolean[""]()
    else:
        raise TypeError("axis must be a boost_histogram or compatible axis")

def convert_storage_type(storage_type, force_atomic = False):
    #TODO move from strings to cppyy syntax and simplify handling of atomic

    if issubclass(storage_type, bh.storage.Double):
        if force_atomic:
            return "narf::atomic_adaptor<double>"
        else:
            return "double"
    elif issubclass(storage_type, bh.storage.Unlimited):
        raise TypeError("Unlimited storage not supported")
    elif issubclass(storage_type, bh.storage.Int64):
        if force_atomic:
            return "narf::atomic_adaptor<std::int64_t>"
        else:
            return "std::int64_t"
    elif issubclass(storage_type, bh.storage.AtomicInt64):
        return "boost::histogram::accumulators::count<std::int64_t, true>"
    elif issubclass(storage_type, bh.storage.Weight):
        if force_atomic:
            #return "boost::histogram::accumulators::weighted_sum<narf::atomic_adaptor<double>>"
            return "narf::atomic_adaptor<boost::histogram::accumulators::weighted_sum<double>>"
        else:
            return "boost::histogram::accumulators::weighted_sum<double>"
    elif issubclass(storage_type, bh.storage.Mean):
        if force_atomic:
            return "narf::atomic_adaptor<boost::histogram::accumulators::mean<double>>"
        else:
            return "boost::histogram::accumulators::mean<double>"
    elif issubclass(storage_type, bh.storage.WeightedMean):
        if force_atomic:
            return "narf::atomic_adaptor<boost::histogram::accumulators::weighted_mean<double>>"
        else:
            return "boost::histogram::accumulators::weighted_mean<double>"
    else:
        raise TypeError("storage must be a boost_histogram or compatible storage type")

def make_array_interface_view(boost_hist):
    view = boost_hist.view(flow = True)
    addr = view.__array_interface__["data"][0]
    arr = cppyy.ll.reinterpret_cast["void*"](addr)

    elem_size = int(view.__array_interface__["typestr"][2:])
    shape = view.__array_interface__["shape"]
    strides = view.__array_interface__["strides"]

    # compute strides for a fortran-style contiguous array with the given shape
    # TODO factorize this into a common function
    stridesf = []
    current_stride = elem_size
    for axis_size in shape:
        stridesf.append(current_stride)
        current_stride *= axis_size
    stridesf = tuple(stridesf)

    if strides is None:
        #default stride for C-style contiguous array
        strides = tuple(reversed(stridesf))

    acc_type = convert_storage_type(boost_hist._storage_type)
    arrview = ROOT.narf.array_interface_view[acc_type, len(shape)](arr, shape, strides)

    return arrview

def hist_to_pyroot_boost(hist_hist, tensor_rank = 0, force_atomic = False):
    arrview = make_array_interface_view(hist_hist)

    if tensor_rank > 0:
        python_axes = hist_hist.axes[:-tensor_rank]
        tensor_axes = hist_hist.axes[-tensor_rank:]
        tensor_sizes = []
        for axis in tensor_axes:
            if axis.traits.underflow:
                raise ValueError("Tensor axes cannot have underflow bins!")
            tensor_sizes.append(axis.size)

        scalar_type = ROOT.double
        dimensions = ROOT.Eigen.Sizes[tuple(tensor_sizes)]

        if issubclass(hist_hist._storage_type, bh.storage.Double):
            cppstoragetype = ROOT.narf.tensor_accumulator[scalar_type, dimensions]
        elif issubclass(hist_hist._storage_type, bh.storage.Weight):
            cppstoragetype = ROOT.narf.tensor_accumulator[ROOT.boost.histogram.accumulators.weighted_sum[scalar_type], dimensions]
        else:
            raise TypeError("Requested storage type is not supported with tensor weights currently")

        if force_atomic:
            cppstoragetype = ROOT.narf.atomic_adaptor[cppstoragetype]
    else:
        python_axes = hist_hist.axes
        cppstoragetype = convert_storage_type(hist_hist._storage_type, force_atomic = force_atomic)

    cppaxes = [ROOT.std.move(convert_axis(axis)) for axis in python_axes]

    pyroot_boost_hist = ROOT.narf.make_histogram_dense[cppstoragetype](*cppaxes)

    arrview.to_boost(pyroot_boost_hist)

    return pyroot_boost_hist

def _histo_boost(df, name, axes, cols, storage = bh.storage.Weight(), force_atomic = ROOT.ROOT.IsImplicitMTEnabled(), tensor_axes = None):
    # first construct a histogram from the hist python interface, then construct a boost histogram
    # using PyROOT with compatible axes and storage types, adopting the underlying storage
    # of the python hist histogram

    #TODO this code can be shared with root histogram version

    coltypes = [df.GetColumnType(col) for col in cols]

    has_weight = len(cols) == (len(axes) + 1)
    tensor_weight = False
    python_axes = axes.copy()

    if has_weight:
        traits = ROOT.narf.tensor_traits[coltypes[-1]]
        if traits.is_tensor:
            tensor_weight = True
            # weight is a tensor-type, use optimized storage and use or create additional axes
            # corresponding to tensor indices
            if tensor_axes is None:
                tensor_axes = [None]*traits.rank

            for idim, (size, tensor_axis) in enumerate(zip(traits.get_sizes(), tensor_axes)):
                if isinstance(tensor_axis, bh.axis.Axis):
                    if tensor_axis.traits.underflow:
                        raise ValueError("Tensor axes cannot have underflow bins!")
                    if tensor_axis.size != size:
                        raise ValueError("Tensor axis must have the same size as the corresponding tensor dimension")
                    python_axes.append(tensor_axis)
                elif isinstance(tensor_axis, str):
                    python_axes.append(hist.axis.Integer(0, size, underflow=False, overflow=False, name = tensor_axis))
                elif tensor_axis is None:
                    python_axes.append(hist.axis.Integer(0, size, underflow=False, overflow=False, name = f"tensor_axis_{idim}"))
                else:
                    raise TypeError("Invalid type provided for tensor axis")

    _hist = hist.Hist(*python_axes, storage = storage, name = name)

    arrview = make_array_interface_view(_hist)

    if tensor_weight:
        # weight is a tensor type, using tensor-storage directly
        tensor_type = coltypes[-1]
        # convert from string to cppyy type
        tensor_type = ROOT.narf.type_identity[tensor_type].type

        if isinstance(storage, bh.storage.Double):
            cppstoragetype = ROOT.narf.tensor_accumulator[tensor_type.Scalar, tensor_type.Dimensions]
        elif isinstance(storage, bh.storage.Weight):
            cppstoragetype = ROOT.narf.tensor_accumulator[ROOT.boost.histogram.accumulators.weighted_sum[tensor_type.Scalar], tensor_type.Dimensions]
        else:
            raise TypeError("Requested storage type is not supported with tensor weights currently")

        if force_atomic:
            cppstoragetype = ROOT.narf.atomic_adaptor[cppstoragetype]
    else:
        cppstoragetype = convert_storage_type(type(storage), force_atomic = force_atomic)

    cppaxes = [ROOT.std.move(convert_axis(axis)) for axis in axes]



    #print("storage type: ", cppstoragetype.__cpp_name__)
    hfill = ROOT.narf.make_histogram_dense[cppstoragetype](*cppaxes)

    #if arrview.size() != hfill.size():
        #raise ValueError("Mismatched sizes")

    #ROOT.gInterpreter.Declare(f"template class narf::FillBoostHelperAtomic<{type(h).__cpp_name__}, {type(hfill).__cpp_name__}>;")

    helper = ROOT.narf.FillBoostHelperAtomic[type(arrview), type(hfill)](ROOT.std.move(arrview), ROOT.std.move(hfill))

    targs = tuple([type(df), type(helper)] + coltypes)

    #if tensor_weight:
        #targsnames = [type(df).__cpp_name__, type(helper).__cpp_name__] + coltypes
        #targsstr = ",".join(targsnames)
        #functemplate = f"template ROOT::RDF::RResultPtr<{type(arrview).__cpp_name__}> narf::book_helper<{targsstr}>({type(df).__cpp_name__}&, {type(helper).__cpp_name__}&&, const std::vector<std::string>&);"
        ##ROOT.gInterpreter.Declare(f"template ROOT::RDF::RResultPtr<{type(arrview).__cpp_name__}> narf::book_helper<{targsstr}>({type(df).__cpp_name__}&, {type(helper).__cpp_name__}&&, const std::vector<std::string>&);")
        #print(name)
        #ROOT.gInterpreter.Declare(functemplate)
        ##print(functemplate)
        ##assert(0)

    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)

    res._hist = _hist

    # hide underlying C++ class and return the python version instead

    res._GetPtr = res.GetPtr

    def get_hist():
        res._GetPtr()
        return res._hist

    def hist_getitem(*args, **kwargs):
        res._GetPtr()
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

def _convert_root_axis_to_hist(axis, name=None):
    is_regular = axis.GetXbins().fN == 0

    if is_regular:
        nbins = axis.GetNbins()
        xlow = axis.GetXmin()
        xhigh = axis.GetXmax()
        return hist.axis.Regular(nbins, xlow, xhigh, name=name) if name \
                else hist.axis.Regular(nbins, xlow, xhigh)
    else:
        nbins = axis.GetNbins()
        edges = [edge for edge in axis.GetXbins()]
        # check if this is actually regular
        edges = np.array(edges, dtype=np.float64)
        if nbins > 1:
            maxdiff2 = np.max(np.abs(np.diff(edges, n = 2)))
            is_regular = maxdiff2/(edges[-1] - edges[0]) < 1e-9
        else:
            is_regular = True
        if is_regular:
            return hist.axis.Regular(nbins, edges[0], edges[-1], name=name) if name else \
              hist.axis.Regular(nbins, edges[0], edges[-1])
        else:
            return hist.axis.Variable(edges, name=name) if name else \
                hist.axis.Variable(edges)

def root_to_hist(root_hist, axis_names=None):
    axes = []
    if isinstance(root_hist, ROOT.TH3):
        axes.append(root_hist.GetXaxis())
        axes.append(root_hist.GetYaxis())
        axes.append(root_hist.GetZaxis())
    elif isinstance(root_hist, ROOT.TH2):
        axes.append(root_hist.GetXaxis())
        axes.append(root_hist.GetYaxis())
    elif isinstance(root_hist, ROOT.TH1):
        axes.append(root_hist.GetXaxis())
    elif isinstance(root_hist, ROOT.THnBase):
        for axis in root_hist.GetListOfAxes():
            axes.append(axis)

    if not axis_names:
        axis_names = [None for a in axes]
    if len(axes) != len(axis_names):
        raise ValueError(f"Number of names given ({len(axis_names)}) does not match number of axes ({len(axes)})")

    boost_axes = [_convert_root_axis_to_hist(axis, name) for axis, name in zip(axes, axis_names)]
    boost_hist = hist.Hist(*boost_axes, storage = bh.storage.Weight())

    view = boost_hist.view(flow = True)
    addr = view.__array_interface__["data"][0]
    arr = cppyy.ll.reinterpret_cast["void*"](addr)

    elem_size = int(view.__array_interface__["typestr"][2:])
    shape = view.__array_interface__["shape"]
    strides = view.__array_interface__["strides"]

    # compute strides for a fortran-style contiguous array with the given shape
    # TODO factorize this into a common function
    stridesf = []
    current_stride = elem_size
    for axis_size in shape:
        stridesf.append(current_stride)
        current_stride *= axis_size
    stridesf = tuple(stridesf)

    if strides is None:
        #default stride for C-style contiguous array
        strides = tuple(reversed(stridesf))

    acc_type = convert_storage_type(boost_hist._storage_type)
    arrview = ROOT.narf.array_interface_view[acc_type, len(shape)](arr, shape, strides)

    arrview.from_root(root_hist)

    return boost_hist


def hist_to_root(boost_hist):
    is_variable = False
    for axis in boost_hist.axes:
        if isinstance(axis, bh.axis.Variable) or (isinstance(axis, bh.axis.Regular) and axis.transform is not None):
            is_variable = True

    name = str(uuid.uuid1())


    if is_variable:
        nbins = []
        edges = []
        for axis in boost_hist.axes:
            if isinstance(axis, bh.axis.Regular) or isinstance(axis, bh.axis.Variable):
                nbins.append(axis.size)
                edges.append(axis.edges)
            elif isinstance(axis, bh.axis.Integer):
                nbins.append(axis.size)
                edges.append(np.array(axis.edges, dtype=np.float64) - 0.5)
            elif isinstance(axis, bh.axis.Boolean):
                nbins.append(2)
                edges.append([-0.5, 0.5, 1.5])
            else:
                raise TypeError("invalid axis type")

        if len(boost_hist.axes) == 1:
            root_hist = ROOT.TH1D(name, "", nbins[0], edges[0])
        elif len(boost_hist.axes) == 2:
            root_hist = ROOT.TH2D(name, "", nbins[0], edges[0], nbins[1], edges[1])
        elif len(boost_hist.axes) == 3:
            root_hist = ROOT.TH3D(name, "", nbins[0], edges[0], nbins[1], edges[1], nbins[2], edges[2])
        else:
            nbins = array.array("i", nbins)
            root_hist = ROOT.THnT["double"](name, "", len(boost_hist.axes), nbins, edges)
    else:
        nbins = []
        xlows = []
        xhighs = []
        for axis in boost_hist.axes:
            if isinstance(axis, bh.axis.Regular):
                if axis.transform is not None:
                    raise ValueError("transforms are not supported (should never reach here)")

                nbins.append(axis.size)
                xlows.append(axis.edges[0])
                xhighs.append(axis.edges[-1])
            elif isinstance(axis, bh.axis.Integer):
                ilow = axis.bin(0)
                ihigh = axis.bin(axis.size - 1) + 1

                nbins.append(axis.size)
                xlows.append(float(ilow) - 0.5)
                xhighs.append(float(ihigh) - 0.5)
            elif isinstance(axis, bh.axis.Boolean):
                nbins.append(2)
                xlows.append(-0.5)
                xhighs.append(1.5)
            else:
                raise TypeError("invalid axis type")

        if len(boost_hist.axes) == 1:
            root_hist = ROOT.TH1D(name, "", nbins[0], xlows[0], xhighs[0])
        elif len(boost_hist.axes) == 2:
            root_hist = ROOT.TH2D(name, "", nbins[0], xlows[0], xhighs[0],
                                                        nbins[1], xlows[1], xhighs[1])
        elif len(boost_hist.axes) == 3:
            root_hist = ROOT.TH3D(name, "", nbins[0], xlows[0], xhighs[0],
                                                        nbins[1], xlows[1], xhighs[1],
                                                        nbins[2], xlows[2], xhighs[2])
        else:
            nbins = array.array("i", nbins)
            xlows = array.array("d", xlows)
            xhighs = array.array("d", xhighs)
            root_hist = ROOT.THnT["double"](name, "", len(boost_hist.axes), nbins, xlows, xhighs)

    view = boost_hist.view(flow = True)
    addr = view.__array_interface__["data"][0]
    arr = cppyy.ll.reinterpret_cast["void*"](addr)

    elem_size = int(view.__array_interface__["typestr"][2:])
    shape = view.__array_interface__["shape"]
    strides = view.__array_interface__["strides"]

    # compute strides for a fortran-style contiguous array with the given shape
    # TODO factorize this into a common function
    stridesf = []
    current_stride = elem_size
    for axis_size in shape:
        stridesf.append(current_stride)
        current_stride *= axis_size
    stridesf = tuple(stridesf)

    if strides is None:
        #default stride for C-style contiguous array
        strides = tuple(reversed(stridesf))

    acc_type = convert_storage_type(boost_hist._storage_type)
    arrview = ROOT.narf.array_interface_view[acc_type, len(shape)](arr, shape, strides)

    if ROOT.narf.acc_traits[acc_type].is_weighted_sum:
        root_hist.Sumw2()

    arrview.to_root(root_hist)

    return root_hist


def _convert_root_axis_info(nbins, xlow, xhigh, edges):

    if edges:
        return ROOT.boost.histogram.axis.variable[""](edges)
    else:
        return ROOT.boost.histogram.axis.regular[""](nbins, xlow, xhigh)

def _histo_with_boost(df, model, cols):

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

    coltypes = [df.GetColumnType(col) for col in cols]

    has_weight = len(cols) == (len(axes_info) + 1)
    tensor_weight = False

    if has_weight:
        traits = ROOT.narf.tensor_traits[coltypes[-1]]
        if traits.is_tensor:
            # weight is a tensor-type, use optimized storage and create additional axes
            # corresponding to tensor indices
            tensor_weight = True

            is_regular = True
            for axis_info in axes_info:
                if axis_info[3]:
                    is_regular = False

            for size in traits.get_sizes():
                if is_regular:
                    axes_info.append((size, -0.5, float(size) - 0.5, []))
                else:
                    axes_info.append((size, 0., 64., np.arange(size+1, dtype=np.float64) - 0.5))

            if len(axes_info) == 1:
                hist_type = ROOT.TH1D
                if is_regular:
                    model = ROOT.RDF.TH1DModel(model.fName, model.fTitle,
                                               axes_info[0][0], axes_info[0][1], axes_info[0][2])
                else:
                    model = ROOT.RDF.TH1DModel(model.fName, model.fTitle,
                                               axes_info[0][0], axes_info[0][3])
            elif len(axes_info) == 2:
                hist_type = ROOT.TH2D
                if is_regular:
                    model = ROOT.RDF.TH1DModel(model.fName, model.fTitle,
                                               axes_info[0][0], axes_info[0][1], axes_info[0][2],
                                               axes_info[1][0], axes_info[1][1], axes_info[1][2])
                else:
                    model = ROOT.RDF.TH2DModel(model.fName, model.fTitle,
                                               axes_info[0][0], axes_info[0][3],
                                               axes_info[1][0], axes_info[1][3])
            elif len(axes_info) == 3:
                hist_type = ROOT.TH2D
                if is_regular:
                    model = ROOT.RDF.TH1DModel(model.fName, model.fTitle,
                                               axes_info[0][0], axes_info[0][1], axes_info[0][2],
                                               axes_info[1][0], axes_info[1][1], axes_info[1][2],
                                               axes_info[2][0], axes_info[2][1], axes_info[2][2])
                else:
                    model = ROOT.RDF.TH2DModel(model.fName, model.fTitle,
                                               axes_info[0][0], axes_info[0][3],
                                               axes_info[1][0], axes_info[1][3],
                                               axes_info[2][0], axes_info[2][3])
            else:
                hist_type = hist_type = ROOT.THnT["double"]
                if is_regular:
                    model = ROOT.RDF.THnDModel(model.fName, model.fTitle, len(axes_info),
                                               [axis_info[0] for axis_info in axes_info],
                                               [axis_info[1] for axis_info in axes_info],
                                               [axis_info[2] for axis_info in axes_info])
                else:
                    model = ROOT.RDF.THnDModel(model.fName, model.fTitle, len(axes_info),
                                               [axis_info[0] for axis_info in axes_info],
                                               [axis_info[3] for axis_info in axes_info])



    if tensor_weight:
        tensor_type = coltypes[-1]
        # convert from string to cppyy type
        tensor_type = ROOT.narf.type_identity[tensor_type].type
        cppstoragetype = ROOT.narf.tensor_accumulator[ROOT.boost.histogram.accumulators.weighted_sum[tensor_type.Scalar], tensor_type.Dimensions]
    else:
        cppstoragetype = ROOT.boost.histogram.accumulators.weighted_sum[ROOT.double]

    if ROOT.ROOT.IsImplicitMTEnabled():
        cppstoragetype = ROOT.narf.atomic_adaptor[cppstoragetype]

    boost_hist = ROOT.narf.make_histogram_dense[cppstoragetype](*boost_axes)

    helper = ROOT.narf.FillBoostHelperAtomic[hist_type, type(boost_hist)](model, ROOT.std.move(boost_hist))

    targs = tuple([type(df), type(helper)] + coltypes)

    #targsnames = [type(df).__cpp_name__, type(helper).__cpp_name__] + coltypes
    #targsstr = ",".join(targsnames)
    #ROOT.gInterpreter.Declare(f"template ROOT::RDF::RResultPtr<{hist_type.__cpp_name__}> narf::book_helper<{targsstr}>({type(df).__cpp_name__}&, {type(helper).__cpp_name__}&&, const std::vector<std::string>&);")
    #assert(0)

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
