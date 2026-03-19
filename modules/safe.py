import pickle
import collections
import torch
import numpy
import _codecs
import zipfile
import re

from modules import errors

TypedStorage = torch.storage.TypedStorage if hasattr(torch.storage, 'TypedStorage') else torch.storage._TypedStorage

def encode(*args):
    return _codecs.encode(*args)
class RestrictedUnpickler(pickle.Unpickler):
    extra_handler = None

    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        try:
            return TypedStorage(_internal=True)
        except TypeError:
            return TypedStorage()

    def find_class(self, module, name):
        if self.extra_handler:
            res = self.extra_handler(module, name)
            if res is not None:
                return res

        allowed = {
            ('collections', 'OrderedDict'): collections.OrderedDict,
            ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
            ('torch._utils', '_rebuild_parameter'): torch._utils._rebuild_parameter,
            ('torch._utils', '_rebuild_device_tensor_from_numpy'): torch._utils._rebuild_device_tensor_from_numpy,
            ('torch', 'FloatStorage'): torch.FloatStorage,
            ('torch', 'HalfStorage'): torch.HalfStorage,
            ('torch', 'IntStorage'): torch.IntStorage,
            ('torch', 'LongStorage'): torch.LongStorage,
            ('torch', 'DoubleStorage'): torch.DoubleStorage,
            ('torch', 'ByteStorage'): torch.ByteStorage,
            ('torch', 'float32'): torch.float32,
            ('torch', 'BFloat16Storage'): torch.BFloat16Storage,
            ('torch.nn.modules.container', 'ParameterDict'): torch.nn.modules.container.ParameterDict,
            ('numpy.core.multiarray', 'scalar'): numpy.core.multiarray.scalar,
            ('numpy.core.multiarray', '_reconstruct'): numpy.core.multiarray._reconstruct,
            ('numpy', 'dtype'): numpy.dtype,
            ('numpy', 'ndarray'): numpy.ndarray,
            ('_codecs', 'encode'): encode,
            ('__builtin__', 'set'): set,
        }

        if (module, name) in allowed:
            return allowed[(module, name)]

        if module == "pytorch_lightning.callbacks" and name == 'model_checkpoint':
            import pytorch_lightning.callbacks
            return pytorch_lightning.callbacks.model_checkpoint
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == 'ModelCheckpoint':
            import pytorch_lightning.callbacks.model_checkpoint
            return pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint

        raise Exception(f"global '{module}/{name}' is forbidden")
allowed_zip_names_re = re.compile(r"^([^/]+)/((data/\d+)|version|byteorder|.data/serialization_id|(data\.pkl))$")
data_pkl_re = re.compile(r"^([^/]+)/data\.pkl$")

def check_zip_filenames(filename, names):
    for name in names:
        if not allowed_zip_names_re.match(name):
            raise Exception(f"bad file inside {filename}: {name}")
def check_pt(filename, extra_handler):
    with zipfile.ZipFile(filename) as z:
        check_zip_filenames(filename, z.namelist())

        for f in z.namelist():
            if data_pkl_re.match(f):
                with z.open(f) as file:
                    unpickler = RestrictedUnpickler(file)
                    unpickler.extra_handler = extra_handler
                    unpickler.load()
                return

        raise Exception(f"data.pkl not found in {filename}")
unsafe_torch_load = torch.load
global_extra_handler = None

def load(filename, *args, **kwargs):
    return load_with_extra(filename, *args, extra_handler=global_extra_handler, **kwargs)

def load_with_extra(filename, extra_handler=None, *args, **kwargs):
    from modules import shared

    if shared.cmd_opts.disable_safe_unpickle:
        return unsafe_torch_load(filename, *args, **kwargs)

    try:
        check_pt(filename, extra_handler)
    except Exception:
        errors.report(
            f"Error verifying pickled file from {filename}\n"
            f"File may be corrupted or malicious. Use --disable-safe-unpickle to skip check.\n",
            exc_info=True,
        )
        return None

    return unsafe_torch_load(filename, *args, **kwargs)
class Extra:
    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        global global_extra_handler
        assert global_extra_handler is None, 'already inside an Extra() block'
        global_extra_handler = self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        global global_extra_handler
        global_extra_handler = None
