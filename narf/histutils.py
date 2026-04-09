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
import narf.clingutils
import awkward as ak
import narf.rdfutils

narf.clingutils.Declare('#include "histutils.hpp"')
narf.clingutils.Declare('#include "FillBoostHelperAtomic.hpp"')

narf.clingutils.Declare('#include <eigen3/Eigen/Dense>')
narf.clingutils.Declare('#include <eigen3/unsupported/Eigen/CXX11/Tensor>')

class SparseStorage:
    """Storage option for HistoBoost selecting a narf::concurrent_sparse_storage
    backed by a narf::concurrent_flat_map. Conversion to a python hist.Hist
    object is not supported in this mode.

    Parameters
    ----------
    fill_fraction : float
        Estimated fraction of bins (including under/overflow) that will be
        populated. Used to size the underlying concurrent_flat_map so that
        most fills hit the initial allocation rather than triggering
        on-the-fly expansion. Values outside (0, 1] are accepted; pass a
        small number for very sparse fills.
    """
    def __init__(self, fill_fraction=0.1):
        self.fill_fraction = float(fill_fraction)


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
    strides = view.strides
        
    underflow = [axis.traits.underflow for axis in boost_hist.axes]

    acc_type = convert_storage_type(boost_hist.storage_type)
    arrview = ROOT.narf.array_interface_view[acc_type, len(shape)](arr, shape, strides, underflow)

    return arrview

def hist_to_pyroot_boost(hist_hist, tensor_rank = 0, force_atomic = False):
    arrview = make_array_interface_view(hist_hist)

    if tensor_rank > 0:
        python_axes = hist_hist.axes[:-tensor_rank]
        tensor_axes = hist_hist.axes[-tensor_rank:]
        tensor_sizes = []
        for axis in tensor_axes:
            tensor_sizes.append(axis.size)

        scalar_type = ROOT.double
        dimensions = ROOT.Eigen.Sizes[tuple(tensor_sizes)]

        if issubclass(hist_hist.storage_type, bh.storage.Double):
            cppstoragetype = ROOT.narf.tensor_accumulator[scalar_type, dimensions]
        elif issubclass(hist_hist.storage_type, bh.storage.Weight):
            cppstoragetype = ROOT.narf.tensor_accumulator[ROOT.boost.histogram.accumulators.weighted_sum[scalar_type], dimensions]
        else:
            raise TypeError("Requested storage type is not supported with tensor weights currently")

        if force_atomic:
            cppstoragetype = ROOT.narf.atomic_adaptor[cppstoragetype]
    else:
        python_axes = hist_hist.axes
        cppstoragetype = convert_storage_type(hist_hist.storage_type, force_atomic = force_atomic)

    cppaxes = [ROOT.std.move(convert_axis(axis)) for axis in python_axes]

    pyroot_boost_hist = ROOT.narf.make_histogram_dense[cppstoragetype](*cppaxes)

    arrview.to_boost(pyroot_boost_hist)

    return pyroot_boost_hist

def _histo_boost(df, name, axes, cols, storage = bh.storage.Weight(), force_atomic = None, tensor_axes = None, convert_to_hist = True):
    # first construct a histogram from the hist python interface, then construct a boost histogram
    # using PyROOT with compatible axes and storage types, adopting the underlying storage
    # of the python hist histogram

    if force_atomic is None:
        force_atomic = ROOT.ROOT.IsImplicitMTEnabled()

    # Sparse storage path: build a narf::sparse_histogram backed by a
    # concurrent_flat_map. The result is exposed as a wums.SparseHist.
    if isinstance(storage, SparseStorage):
        if tensor_axes is not None:
            raise NotImplementedError("Tensor weights are not supported with SparseStorage")
        coltypes = [df.GetColumnType(col) for col in cols]
        for coltype in coltypes[len(axes):]:
            traits = ROOT.narf.tensor_traits[coltype]
            if traits.is_tensor:
                raise NotImplementedError("Tensor weights are not supported with SparseStorage")
        cppaxes = [ROOT.std.move(convert_axis(axis)) for axis in axes]
        hfill = ROOT.narf.make_histogram_sparse[ROOT.narf.atomic_adaptor[ROOT.double]](storage.fill_fraction, *cppaxes)
        helper = ROOT.narf.FillBoostHelperAtomic[type(hfill)](ROOT.std.move(hfill))
        targs = tuple([type(df), type(helper)] + coltypes)
        res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)

        if not convert_to_hist:
            return res

        # Lazily convert the underlying C++ sparse histogram to a wums.SparseHist
        # the first time the result is dereferenced.
        from wums.sparse_hist import SparseHist

        res._GetPtr = res.GetPtr
        res._sparse_hist = None
        python_axes_sparse = list(axes)

        def _build_sparse():
            if res._sparse_hist is not None:
                return res._sparse_hist
            cpp_hist = res._GetPtr()
            snapshot = ROOT.narf.sparse_histogram_snapshot(cpp_hist)
            n = len(snapshot)
            boost_flat = np.empty(n, dtype=np.int64)
            vals = np.empty(n, dtype=np.float64)
            for i, kv in enumerate(snapshot):
                boost_flat[i] = int(kv.first)
                vals[i] = float(kv.second)
            extents = tuple(int(ax.extent) for ax in python_axes_sparse)
            size = int(np.prod(extents)) if extents else 1
            # boost::histogram linearizes column-major (leftmost axis = stride 1),
            # but wums.SparseHist expects numpy row-major (C order). Remap by
            # un-raveling under F order and re-raveling under C order.
            if n and len(extents) > 1:
                multi = np.unravel_index(boost_flat, extents, order="F")
                flat = np.ravel_multi_index(multi, extents, order="C").astype(np.int64)
            else:
                flat = boost_flat
            res._sparse_hist = SparseHist._from_flat(flat, vals, python_axes_sparse, size)
            return res._sparse_hist

        ret_null = lambda: None
        res.__deref__ = _build_sparse
        res.__follow__ = _build_sparse
        res.begin = ret_null
        res.end = ret_null
        res.GetPtr = _build_sparse
        res.GetValue = _build_sparse
        return res

    #TODO some of this code can be shared with root histogram version

    #FIXME make this more generic
    accumulator_args = 1 if (isinstance(storage, bh.storage.Mean) or isinstance(storage, bh.storage.WeightedMean)) else 0

    if accumulator_args:
        raise NotImplementedError(f"Storage type {type(storage)} takes accumulator arguments, which is not supported currently.")

    nargs = len(axes) + accumulator_args

    if len(cols) < nargs:
        raise RuntimeError(f"Mismatched number of columns and axes for axes {axes}, storage {storage} and columns {cols}")

    coltypes = [df.GetColumnType(col) for col in cols]

    tensor_weight = False
    tensor_scalar_type = None
    tensor_shape = []

    for coltype in coltypes[nargs:]:
        traits = ROOT.narf.tensor_traits[coltype]

        if traits.is_tensor:
            if tensor_weight:
                raise NotImplementedError("Multiple tensor weights are not currently supported.")

            if tensor_weight and traits.tensor_type.Scalar != tensor_scalar_type:
                raise RuntimeError(f"Incompatible tensor types {traits.tensor_type} and {tensor_type}")

            tensor_weight = True
            tensor_scalar_type = traits.tensor_type.Scalar
            # TODO also/instead handle the case that tensor axes are in common and elements should be multiplied directly
            # instead of concatenating the axes?
            tensor_shape.extend(traits.get_sizes())

    python_axes = axes.copy()

    if tensor_weight:
        # weight is a tensor-type, use optimized storage and use or create additional axes
        # corresponding to tensor indices
        if tensor_axes is None:
            tensor_axes = [None]*len(tensor_shape)

        for idim, (size, tensor_axis) in enumerate(zip(tensor_shape, tensor_axes)):
            if isinstance(tensor_axis, bh.axis.Axis):
                if tensor_axis.size != size:
                    raise ValueError("Tensor axis must have the same size as the corresponding tensor dimension")
                python_axes.append(tensor_axis)
            elif isinstance(tensor_axis, str):
                python_axes.append(hist.axis.Integer(0, size, underflow=False, overflow=False, name = tensor_axis))
            elif tensor_axis is None:
                python_axes.append(hist.axis.Integer(0, size, underflow=False, overflow=False, name = f"tensor_axis_{idim}"))
            else:
                raise TypeError("Invalid type provided for tensor axis")

        # weight is a tensor type, using tensor-storage directly
        dimensions = ROOT.Eigen.Sizes[*tensor_shape]

        if isinstance(storage, bh.storage.Double):
            cppstoragetype = ROOT.narf.tensor_accumulator[tensor_scalar_type, dimensions]
        elif isinstance(storage, bh.storage.Weight):
            cppstoragetype = ROOT.narf.tensor_accumulator[ROOT.boost.histogram.accumulators.weighted_sum[tensor_scalar_type], dimensions]
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

    #narf.clingutils.Declare(f"template class narf::FillBoostHelperAtomic<{type(h).__cpp_name__}, {type(hfill).__cpp_name__}>;")


    if convert_to_hist:
        _hist = hist.Hist(*python_axes, storage = storage, name = name)
        arrview = make_array_interface_view(_hist)

        helper = ROOT.narf.FillBoostHelperAtomic[type(arrview), type(hfill)](ROOT.std.move(arrview), ROOT.std.move(hfill))
    else:
        helper = ROOT.narf.FillBoostHelperAtomic[type(hfill)](ROOT.std.move(hfill))


    targs = tuple([type(df), type(helper)] + coltypes)

    #if tensor_weight:
        #targsnames = [type(df).__cpp_name__, type(helper).__cpp_name__] + coltypes
        #targsstr = ",".join(targsnames)
        #functemplate = f"template ROOT::RDF::RResultPtr<{type(arrview).__cpp_name__}> narf::book_helper<{targsstr}>({type(df).__cpp_name__}&, {type(helper).__cpp_name__}&&, const std::vector<std::string>&);"
        ##narf.clingutils.Declare(f"template ROOT::RDF::RResultPtr<{type(arrview).__cpp_name__}> narf::book_helper<{targsstr}>({type(df).__cpp_name__}&, {type(helper).__cpp_name__}&&, const std::vector<std::string>&);")
        #print(name)
        #narf.clingutils.Declare(functemplate)
        ##print(functemplate)
        ##assert(0)

    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)

    if convert_to_hist:

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

    arrview = make_array_interface_view(boost_hist)

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
    
    arrview = make_array_interface_view(boost_hist)
        
    if arrview.is_weighted_sum():
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
    #narf.clingutils.Declare(f"template ROOT::RDF::RResultPtr<{hist_type.__cpp_name__}> narf::book_helper<{targsstr}>({type(df).__cpp_name__}&, {type(helper).__cpp_name__}&&, const std::vector<std::string>&);")
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

def build_quantile_hists(df, cols, condaxes, quantaxes, continuous=False):
    """Build histograms which encode conditional quantiles for the provided variables, to be used with define_quantile_ints.

    When ``continuous=True`` the original quantile axes are kept as-is in the
    returned helper histograms (instead of being replaced by ``Integer`` axes).
    ``define_quantile_ints`` detects this from the axis type and uses the
    continuous quantile helpers, which return CDF-style values in ``[0, 1]``
    obtained by linearly interpolating between the stored quantile edges.
    """

    arraxes = condaxes + quantaxes

    ncond = len(condaxes)
    nquant = len(quantaxes)

    condcols = cols[:ncond]
    quantcols = cols[ncond:]

    if ncond > 0:
        hist_cond = df.HistoBoost("hist_cond", condaxes, condcols, storage=hist.storage.Int64())

    # this triggers an event loop, so it has to go last
    # FIXME ak.from_rdataframe triggers C++ errors when proper error handling is enabled in cppyy so fallback to the manual
    # case for now which is restricted to scalar columns
    try:
        arr = ak.from_rdataframe(df, cols)
    except TypeError:
        arrs = {col : df.Take[df.GetColumnType(col)](col) for col in cols}
        akdict = {col : np.array(arr.GetValue(), copy=False) for col, arr in arrs.items()}
        del arrs
        arr = ak.Array(akdict)

    if ncond > 0:
        hist_cond = hist_cond.GetValue()

    # for axis in hist_cond.axes:
        # print(hist_cond.project(axis.name))

    shape_final = [axis.extent for axis in arraxes]

    arr = arr[None, ...]

    quantile_integer_axes = []
    quantile_hists = []


    for iax, (col, axis) in enumerate(zip(cols, arraxes)):
        sortidxs = ak.argsort(arr[..., col], axis=-1, stable=False)
        arr = arr[sortidxs]

        if col in condcols:
            projnames = [axis.name for axis in condaxes[:iax+1]]
            hcondpartial = hist_cond.project(*projnames)

            quantile_counts = hcondpartial.values(flow=True)
            quantile_counts = np.ravel(quantile_counts)

        else:
            counts = ak.num(arr[..., col], axis=-1)
            quantile_counts_cumulative = axis.edges*counts[:, None]

            quantile_counts_cumulative = ak.values_astype(np.rint(quantile_counts_cumulative), np.int64)
            quantile_counts = quantile_counts_cumulative[..., 1:] - quantile_counts_cumulative[..., :-1]
            quantile_counts = ak.flatten(quantile_counts, axis=None)

        arr = ak.flatten(arr, axis=1)
        arr = ak.unflatten(arr, quantile_counts, axis=0)

        if col in quantcols:
            quantile_edges = ak.max(arr[..., col], axis=-1, mask_identity=False)
            quantile_edges = np.reshape(quantile_edges, shape_final[:iax+1])
            quantile_edges = ak.to_numpy(quantile_edges)

            # replace -infinity from empty values with the previous bin edge
            # so that the quantile edges are at least still monotonic
            nquants = quantile_edges.shape[-1]
            for iquant in range(1, nquants):
                quantile_edges[..., iquant] = np.where(quantile_edges[..., iquant]==-np.inf, quantile_edges[..., iquant-1], quantile_edges[..., iquant])

            iquantax = iax - ncond

            if continuous:
                # Keep the original (Regular / Variable) quantile axis so that
                # subsequent helpers in the chain can be indexed by the
                # continuous CDF-style output of the previous helper.
                quantile_integer_axis = axis
            else:
                quantile_integer_axis = hist.axis.Integer(0, axis.size, underflow=False, overflow=False, name=f"{axis.name}_int")
            quantile_integer_axes.append(quantile_integer_axis)

            helper_axes = condaxes[:iax+1] + quantile_integer_axes

            helper_hist = hist.Hist(*helper_axes)
            helper_hist.values(flow=True)[...] = quantile_edges

            quantile_hists.append(helper_hist)

    return quantile_hists


def define_quantile_ints(df, cols, quantile_hists):
    """Define transformed columns corresponding to conditional quantiles.

    By default the helpers return the integer quantile bin index. If the
    helper histograms produced by :func:`build_quantile_hists` were built in
    continuous mode (their trailing quantile axis is not a plain ``Integer``
    axis), the continuous quantile helpers are used instead, returning a
    CDF-style value in ``[0, 1]``.
    """

    ncols = len(cols)
    nquant = len(quantile_hists)
    ncond = ncols - nquant

    condcols = cols[:ncond]
    quantcols = cols[ncond:]

    # Detect continuous mode from the trailing (quantile) axis of the first
    # helper histogram: continuous-mode helper histograms preserve the
    # original (Regular / Variable) quantile axis, integer-mode ones use a
    # generated Integer axis.
    continuous = not isinstance(quantile_hists[0].axes[-1], hist.axis.Integer)

    helper_cols_cond = condcols.copy()

    suffix = "_quant" if continuous else "_iquant"

    for col, quantile_hist in zip(quantcols, quantile_hists):

        if len(quantile_hist.axes) > 1:
            helper_hist = narf.hist_to_pyroot_boost(quantile_hist, tensor_rank=1)
            if continuous:
                quanthelper = ROOT.narf.make_quantile_helper_continuous(ROOT.std.move(helper_hist))
            else:
                quanthelper = ROOT.narf.make_quantile_helper(ROOT.std.move(helper_hist))
        else:
            # special case for static quantiles with no conditional variables
            vals = quantile_hist.values()
            arr = ROOT.std.array["double", vals.size](vals)
            if continuous:
                quanthelper = ROOT.narf.QuantileHelperStaticContinuous[vals.size](arr)
            else:
                quanthelper = ROOT.narf.QuantileHelperStatic[vals.size](arr)

        helper_cols = helper_cols_cond + [col]

        outname = f"{col}{suffix}"
        df = narf.rdfutils.flexible_define(df, outname, quanthelper, helper_cols)
        helper_cols_cond.append(outname)

    quantile_axes = list(quantile_hists[-1].axes)
    quantile_cols = helper_cols_cond

    return df, quantile_axes, quantile_cols

def shifted_smeared_hist_weight(df, name, axes, original_cols, shifted_cols=None, smear_shifted_cols=None, nominal_weight_col=None):
    cppaxes = [ROOT.std.move(convert_axis(axis)) for axis in axes]
    helper = ROOT.narf.make_hist_shift_helper(*cppaxes)

    if shifted_cols is None:
        shifted_cols = original_cols

    if smear_shifted_cols is None:
        smear_shifted_cols = original_cols

    helper_cols = original_cols + shifted_cols + smear_shifted_cols

    if nominal_weight_col:
        helper_cols += [nominal_weight_col]

    return narf.rdfutils.flexible_define(df, name, ROOT.std.move(helper), helper_cols)

def shifted_hist(df, name, axes, original_cols, shifted_cols=None, smear_shifted_cols=None, nominal_weight_col=None):
    shifted_weight_name = f"{name}_shifted_weight"

    df_tmp = shifted_smeared_hist_weight(df, shifted_weight_name, axes, original_cols, shifted_cols, smear_shifted_cols, nominal_weight_col)

    hist_cols = original_cols + [shifted_weight_name]

    h = df_tmp.HistoBoost(name, axes, hist_cols)
    return h

    # return rdfutils


