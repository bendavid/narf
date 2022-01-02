import ROOT
import boost_histogram as bh
import hist
from hashlib import sha256
import time
import math
import cppyy.ll

ROOT.gInterpreter.Declare('#include "histutils.h"')
ROOT.gInterpreter.Declare('#include "FillBoostHelperAtomic.h"')

#TODO add metadata?

def convert_axis(axis):
    default = ROOT.boost.histogram.use_default

    #TODO make this less implementation dependent
    options = 0
    if (axis.traits.underflow):
        options |= 1
    if (axis.traits.overflow):
        options |= 2
    if (axis.traits.circular):
        options |= 4
    if (axis.traits.growth):
        options |= 8

    options = ROOT.boost.histogram.axis.option.bitset[options]

    #TODO add metadata support?
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
        ihigh = axis.bin(axis.size)
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
        raise TypeError("axis must be a boost_histogram axis")

def _histo_boost(df, name, axes, cols):
    _hist = hist.Hist(*axes, storage = bh.storage.Weight())

    arr = _hist.view(flow=True).__array_interface__

    addr = arr["data"][0]
    size = math.prod(arr["shape"])

    addrptr = cppyy.ll.reinterpret_cast["void*"](addr)
    print(addrptr)

    cppaxes = [convert_axis(axis) for axis in axes]
    h = ROOT.narf.make_atomic_histogram_with_error_adopted(addrptr, size, *cppaxes)

    helper = ROOT.narf.FillBoostHelperAtomic[type(h)](ROOT.std.move(h))
    coltypes = [df.GetColumnType(col) for col in cols]
    targs = tuple([type(df), type(helper)] + coltypes)
    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)

    res.name = name
    res._hist = _hist
    return res

def _histo_boost_raw(df, axes, cols):
    h = ROOT.narf.make_atomic_histogram_with_error(*axes)
    #h = ROOT.narf.make_histogram_with_error(*axes)
    #h = ROOT.narf.make_atomic_histogram(*axes)
    helper = ROOT.narf.FillBoostHelperAtomic[type(h)](ROOT.std.move(h))
    coltypes = [df.GetColumnType(col) for col in cols]
    targs = tuple([type(df), type(helper)] + coltypes)
    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)
    return res

def _ret_null(resultptr):
    return None

def _make_hist(result_ptr):
    print("calling conversion", result_ptr)

    if result_ptr._hist is not None:
        return

    print("starting conversion")

    print("getting cpp hist")
    cpphist = result_ptr._GetValue()
    print("constructing pythong hist")
    result_ptr._hist = hist.Hist(*result_ptr._axes, storage = bh.storage.Weight())

    print("buffer stuff")

    vals = result_ptr._hist.values(flow=True)
    valsaddr = vals.__array_interface__["data"][0]
    valsstrides = vals.__array_interface__["strides"]

    variances = result_ptr._hist.variances(flow=True)
    varsaddr = variances.__array_interface__["data"][0]
    varsstrides = variances.__array_interface__["strides"]

    print("starting buffer fill")

    ROOT.narf.fill_buffer(cpphist, valsaddr, varsaddr, valsstrides, varsstrides)

    print("done conversion")

def _get_hist(result_ptr):
    result_ptr._GetValue()
    return result_ptr._hist

def _hist_getitem(result_ptr, *args, **kwargs):
    result_ptr._GetValue()
    return result_ptr._hist.__getitem__(*args, **kwargs)

@ROOT.pythonization("RInterface<", ns="ROOT::RDF", is_prefix=True)
def pythonize_rdataframe(klass):
    print("doing RDF pythonization")
    klass.HistoBoost = _histo_boost

@ROOT.pythonization("RResultPtr<", ns="ROOT::RDF", is_prefix=True)
def pythonize_resultptr(klass):
    print("pythonize RResultPtr")

    print(klass.__cpp_name__)

    name = klass.__cpp_name__[len("ROOT::RDF::RResultPtr<"):]
    if not name.startswith("boost::histogram::histogram"):
        return

    print("doing boost histo specific stuff")

    klass._GetValue = klass.GetValue

    klass.__deref__ = _get_hist
    klass.__follow__ = _get_hist
    klass.begin = _ret_null
    klass.end = _ret_null
    klass.GetPtr = _get_hist
    klass.GetValue = _get_hist
    klass.__getitem__ = _hist_getitem



@ROOT.pythonization("histogram<", ns="boost::histogram", is_prefix=True)
def pythonize_boosthist(klass):
    print("doing boost histogram pythonization")


    #fname = sha256(klass.__cpp_name__.encode("utf-8")).hexdigest() + ".cpp"
    #fcontents = f"""
    ##include "histutils.h"
    #//template class {klass.__cpp_name__};
    #{klass.__cpp_name__} test;
    #"""

    #print("class state:", ROOT.TClass.GetClass(klass.__cpp_name__).GetState())

    #with open(fname, "w") as text_file:
        #text_file.write(fcontents)
    #ROOT.gSystem.CompileMacro(fname)
    #ROOT.narf.set_custom_streamer[klass]()

    #print("class state:", ROOT.TClass.GetClass(klass.__cpp_name__).GetState())

