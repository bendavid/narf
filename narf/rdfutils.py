import narf.clingutils
import ROOT

narf.clingutils.Declare('#include "rdfutils.hpp"')
narf.clingutils.Declare('#include "progresshelper.hpp"')

def flexible_graph_operation(df, name, graph_function, func, cols):
    coltypes = [df.GetColumnType(col) for col in cols]
    print(func)
    print(coltypes)
    wrappedhelper = ROOT.narf.DefineWrapper[type(func), *coltypes](func)
    return graph_function(name, wrappedhelper, cols)

def flexible_define(df, name, func, cols):
    return flexible_graph_operation(df, name, df.Define, func, cols)

def flexible_filter(df, name, func, cols):
    return flexible_graph_operation(df, name, df.Filter, func, cols)
