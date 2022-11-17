import ROOT
import pathlib

ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")

from .graph_builder import build_and_run
from .dataset import Dataset
from .histutils import hist_to_root, root_to_hist, hist_to_pyroot_boost
from .fitutils import fit_hist
from .fitutilsjax import fit_hist_jax

__all__ = ["build_and_run", "Dataset", "hist_to_root", "root_to_hist", "hist_to_pyroot_boost", "fit_hist", "fit_hist_jax"]
