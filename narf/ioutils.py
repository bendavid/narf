import pickle
import hdf5plugin
import h5py
import boost_histogram as bh
import hist
import copy
import numpy as np
import uuid
import copyreg
import io
import datetime
import subprocess
import os, sys
import re
import pathlib

MIN_PROTOCOL_VERSION = 1
CURRENT_PROTOCOL_VERSION = 1

class H5PickleProxy:
    """Allows storage of objects in h5py files with lazy reading."""
    def __init__(self, obj, h5group = None):
        self.obj = obj
        self.h5group = h5group

    def get(self):
        if self.obj is None:
            if self.h5group is None:
                raise ValueError("Trying to read object from H5PickleProxy but neither object nor underlying hdf5 storage is available.")
            elif not self.h5group:
                raise ValueError("Trying to read object from H5PickleProxy but the underlying file has been closed.")

            self.obj = pickle_load_h5py(self.h5group)

        return self.obj

    def release(self):
        self.obj = None

class H5Buffer():
    """Signals out-of-band storage of buffers in h5py files during pickling.

    The corresponding data can later be read back directly from the file using readinto.
    The provides an efficient mechanism for restoring objects which are not able to adopt
    external buffers.
    """

    def __init__(self, buf, h5dset = None, readonly = None):
        self.buf = buf
        self.h5dset = h5dset
        self.readonly = readonly

        if self.buf is not None:
            self.readonly = memoryview(self.buf).readonly

    def checkdset(self):
        if self.h5dset is None:
            raise ValueError("Trying to read from H5Buffer but neither buffer nor dset are available.")
        elif not self.h5dset:
            raise ValueError("Trying to read from H5Buffer but underlying file has been closed.")

    def getbuffer(self):
        if self.buf is None:
            self.checkdset()
            bufarr = np.empty_like(self.h5dset)
            if self.h5dset.size:
                self.h5dset.read_direct(bufarr)
            bufout = bufarr.data
            if self.readonly:
                bufout = bufout.toreadonly()
            self.buf = bufout

        return self.buf

    def readinto(self, b, /):
        # erase shape, strides and format information of the destination buffer
        bufarr = np.frombuffer(pickle.PickleBuffer(b).raw(), dtype = np.ubyte)
        if self.buf is None:
            # no source buffer available, read directly from the h5py dataset
            self.checkdset()
            if self.h5dset.size:
                self.h5dset.read_direct(bufarr)
            else:
                # read_direct doesn't work with empty datasets so fall back to slicing.
                # although this is a null operation it acts as a consistency check on
                # the destination buffer size'
                bufarr[...] = self.h5dset[...]
        else:
            # read from the stored buffer, erasing shape, strides and format information
            bufarr[...] = pickle.PickleBuffer(self.buf).raw()

    def __reduce_ex__(self, protocol):
        if protocol >= 5:
            outbuf = pickle.PickleBuffer(outbuf)
        else:
            if memoryview(self.buf).readonly:
                outbuf = bytes(self.buf)
            else:
                outbuf = bytearray(self.buf)

        return (type(self), (outbuf,))

class H5Path():
    """Lightweight softlink object storing the path to an h5py group or dataset within an hdf5 file."""
    def __init__(self, path):
        self.path = path

# functions below are a workaround for suboptimal pickling of bh.Histogram
# which doesn't properly use the out-of-band-mechanism and which can't adopt external buffers
def get_histogram_view(h):
    """Return a view to the contents of a histogram, suitable for use with PickleBuffer or H5Buffer."""
    return h.view(flow=True)

def make_Histogram(axes, storage, metadata, h5buf):
    """Reconstruct a Histogram given constructor arguments and an io-like object with a readinto method."""
    hist_read = bh.Histogram(*axes, storage=storage, metadata=metadata)
    hist_read_view = get_histogram_view(hist_read)
    h5buf.readinto(hist_read_view)
    return hist_read

def reduce_Histogram(obj):
    """Custom reduction function for Histogram storing the constructor arguments and an H5Buffer of the contents."""
    axes = tuple(obj.axes)
    view = get_histogram_view(obj)
    h5buf = H5Buffer(view)

    return (make_Histogram, (axes, obj.storage_type(), obj.metadata, h5buf) )

def make_Hist(axes, storage, metadata, label, name, h5buf):
    """Reconstruct a Hist given constructor arguments and an io-like object with a readinto method."""
    hist_read = hist.Hist(*axes, storage=storage, metadata=metadata, label=label, name=name)
    hist_read_view = get_histogram_view(hist_read)
    h5buf.readinto(hist_read_view)
    return hist_read

def reduce_Hist(obj):
    """Custom reduction function for Hist storing the constructor arguments and an H5Buffer of the contents."""
    axes = tuple(obj.axes)
    view = get_histogram_view(obj)
    h5buf = H5Buffer(view)

    return (make_Hist, (axes, obj.storage_type(), obj.metadata, obj.label, obj.name, h5buf) )

class H5IO():
    """Provides and io-like interface for reading and writing to a resizable h5py dataset."""
    def __init__(self, name, h5out, mode = "r"):
        if mode == "r":
            self.dset = h5out[name]
        elif mode == "w":
            self.dset = h5out.create_dataset(name, shape=(0,), dtype = np.ubyte, maxshape=(None,), chunks = (1024*1024,), **hdf5plugin.Blosc(cname="lz4"))
        else:
            raise ValueError("Unsupported mode.")
        self.pos = 0

    def write(self, b, /):
        if type(b) is bytes:
            flat_view = memoryview(b)
        else:
            # erase shape, strides and format information
            flat_view = pickle.PickleBuffer(b).raw()

        oldsize = self.dset.size
        nbytes = flat_view.nbytes
        size = oldsize + nbytes
        self.dset.resize((size,))
        self.dset[oldsize:] = flat_view
        return nbytes

    def read(self, size=-1, /):
        oldpos = self.pos
        self.pos += size
        return self.dset[oldpos:self.pos].data

    def readline(self, size=-1, /):
        raise NotImplementedError("readline is not implemented")

    def readinto(self, b, /):
        # erase shape, strides and format information of the destination buffer
        arr = np.frombuffer(pickle.PickleBuffer(b).raw(), dtype = np.ubyte)
        nbytes = arr.nbytes
        oldpos = self.pos
        self.pos += nbytes
        if nbytes:
            self.dset.read_direct(arr, source_sel = np.s_[oldpos:self.pos])
        else:
            # read_direct doesn't work with zero bytes so fall back to slicing.
            # although this is a null operation it acts as a consistency check on
            # the destination buffer size'
            arr[...] = self.dset[...]

        return nbytes

class H5Pickler(pickle.Pickler):
    """Implements pickling to h5py file.

    Includes support for additional out-of-band storage for proxy objects and buffers which need
    to be read back directly from the file.  h5py groups and datasets can also be stored via H5Path softlink"""

    # use custom dispatch table with custom reduce functions for boost_histogram.Histogram and hist.Hist
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[bh.Histogram] = reduce_Histogram
    dispatch_table[hist.hist.Hist] = reduce_Hist

    def __init__(self, h5group, protocol = 5):
        self.h5group = h5group
        self.io = H5IO("pickle_data", h5group, mode = "w")
        super().__init__(self.io, protocol=protocol)

    def reduce_H5PickleProxy(self, proxy):
        """Custom reduce function for H5PickleProxy.

        The proxied object is written out-of-band from the main pickle stream.
        The H5PickleProxy object is read back with only the group object where the proxied object can be lazily read back from.
        """

        PROXIED_OBJECTS_NAME = "proxied_objects"
        current_location = self.h5group

        if PROXIED_OBJECTS_NAME in current_location:
            proxied_objects = current_location[PROXIED_OBJECTS_NAME]
        else:
            proxied_objects = current_location.create_group(PROXIED_OBJECTS_NAME, track_order = True)

        iproxy = len(proxied_objects)
        proxied_group = pickle_dump_h5py(f"proxied_object_{iproxy}", proxy.obj, proxied_objects)

        return (type(proxy), (None, proxied_group))

    def reduce_H5Buffer(self, h5buf):
        """Custom reduce function for H5Buffer.

        The corresponding buffer is written out-of-band from the main pickle stream.
        The H5Buffer object is read back with only the dataset object where the underlying data can
        be read back when needed (and directly into an existing buffer if desired).
        """

        H5BUFFER_GROUP_NAME = "h5_buffers"
        current_location = self.h5group

        if H5BUFFER_GROUP_NAME in current_location:
            h5_buffers = current_location[H5BUFFER_GROUP_NAME]
        else:
            h5_buffers = current_location.create_group(H5BUFFER_GROUP_NAME, track_order = True)

        ibuf = len(h5_buffers)
        flat_view = pickle.PickleBuffer(h5buf.getbuffer()).raw()
        bufdset = h5_buffers.create_dataset(f"h5_buffer_{ibuf}", data = flat_view, chunks=True, **hdf5plugin.Blosc(cname="lz4"))

        return (type(h5buf), (None, bufdset, flat_view.readonly))

    def reducer_override(self, obj):
        if type(obj) is H5PickleProxy:
            return self.reduce_H5PickleProxy(obj)
        elif type(obj) is H5Buffer:
            return self.reduce_H5Buffer(obj)
        else:
            return NotImplemented

    def persistent_id(self, obj):
        """Handle storage of h5py group or dataset as H5Path."""
        if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
            return H5Path(obj.name)
        else:
            return None

class H5Unpickler(pickle.Unpickler):
    """Implements unpickling from an h5py file.

    Includes support for reading back references to h5py groups and datasets stored as H5Path softlinks.
    """

    def __init__(self, h5group):
        self.h5group = h5group
        self.io = H5IO("pickle_data", h5group, mode = "r")
        super().__init__(self.io)

    def persistent_load(self, pid):
        """Handle retrieval of h5py group or dataset via H5Path."""
        if type(pid) is H5Path:
            return self.h5group.file[pid.path]
        else:
            raise pickle.UnpicklingError("unsupported persistent object")

def pickle_dump_h5py(name, obj, h5out):
    """Write an object to a new h5py group which will be created within the provided group."""
    obj_group = h5out.create_group(name)
    try:
        obj_group.attrs["narf_h5py_pickle_protocol_version"] = CURRENT_PROTOCOL_VERSION
        pickler = H5Pickler(obj_group)
        pickler.dump(obj)
    except:
        del obj_group
        raise
    return obj_group

def pickle_load_h5py(h5group):
    """Load an object from the h5py group which contains it."""
    try:
        narf_protocol_version = h5group.attrs["narf_h5py_pickle_protocol_version"]
    except KeyError:
        raise ValueError("h5py dataset does not contain a python object pickled by narf.")

    if narf_protocol_version < MIN_PROTOCOL_VERSION:
        raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, minimum supported version is {MIN_PROTOCOL_VERSION}")
    elif narf_protocol_version > CURRENT_PROTOCOL_VERSION:
        raise ValueError(f"Unsuppported narf protocol version {narf_protocol_version}, maximum supported version is {CURRENT_PROTOCOL_VERSION}")

    unpickler = H5Unpickler(h5group)
    return unpickler.load()

def script_command_to_str(argv, parser_args):
    call_args = np.array(argv[1:], dtype=object)
    match_expr = "|".join(["^-+([a-z]+[1-9]*-*)+"]+([] if not parser_args else [f"^-*{x.replace('_', '.')}" for x in vars(parser_args).keys()]))
    if call_args.size != 0:
        flags = np.vectorize(lambda x: bool(re.match(match_expr, x)))(call_args)
        special_chars = np.vectorize(lambda x: not x.isalnum())(call_args)
        select = ~flags & special_chars
        if np.count_nonzero(select):
            call_args[select] = np.vectorize(lambda x: f"'{x}'")(call_args[select])
    return " ".join([argv[0], *call_args])

def make_meta_info_dict(exclude_diff = 'notebooks', args = None, wd = f"{pathlib.Path(__file__).parent}/../"):
    meta_data = {
        "time" : str(datetime.datetime.now()), 
        "command" : script_command_to_str(sys.argv, args),
        "args": {a: getattr(args,a) for a in vars(args)} if args else {}
    }    
    if subprocess.call(["git", "branch"], cwd=wd, stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) != 0:
        meta_data["git_info"] = {"hash" : "Not a git repository!",
                "diff" : "Not a git repository"}
    else:
        meta_data["git_hash"] = subprocess.check_output(['git', 'log', '-1', '--format="%H"'], cwd=wd, encoding='UTF-8')
        diff_comm = ['git', 'diff']
        if exclude_diff:
            diff_comm.extend(['--', f":!{exclude_diff}"])
        meta_data["git_diff"] = subprocess.check_output(diff_comm, encoding='UTF-8', cwd=wd)

    return meta_data