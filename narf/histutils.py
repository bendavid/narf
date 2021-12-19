import ROOT

ROOT.gInterpreter.Declare('#include "histutils.h"')
ROOT.gInterpreter.Declare('#include "FillBoostHelperAtomic.h"')

def _histo_boost(df, name, axes, cols):
    h = ROOT.narf.make_atomic_histogram_with_error(*axes)
    helper = ROOT.narf.FillBoostHelperAtomic[type(h)](ROOT.std.move(h))
    coltypes = [df.GetColumnType(col) for col in cols]
    targs = tuple([type(df), type(helper)] + coltypes)
    res = ROOT.narf.book_helper[targs](df, ROOT.std.move(helper), cols)
    res.name = name
    return res

@ROOT.pythonization("RInterface<", ns="ROOT::RDF", is_prefix=True)
def pythonize_rdataframe(klass):
    print("doing pythonization")
    klass.HistoBoost = _histo_boost

#@ROOT.pythonization("histogram<", ns="boost::histogram", is_prefix=True)
#def pythonize_boosthist(klass):
    #print("doing pythonization")
    #klass.BoostHistoAtomicWithError = _boost_histo_atomic_with_error
