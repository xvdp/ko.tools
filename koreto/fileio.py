""" @xvdp fileio
methods to deal with files

    get_files(folders, ext, recursive) # recursive find file
    get_images(folders, recursive)
    rndlist(inputs)

    hash_file() # hash file, datasizie, content and optional metadata
    hash_folder() # folder or files
    reversedict() # key with subkey 

"""
from typing import Union, Any, Collection
import os
import os.path as osp
import json
import pickle
import hashlib
import numpy as np
from PIL import Image

from koreto import WITH_TORCH
if WITH_TORCH:
    import torch
    Vector = (np.ndarray, torch.Tensor)
else:
    Vector = np.ndarray

def get_files(folder: Union[str, list, tuple]=".", ext: Union[str, list, tuple]=None,
              recursive: bool=False) -> list:
    """ conditional file getter
    Args
        folder      (str|list ['.'])  folder | list of folders
        ext         (str|list [None]) file extensions, default, any
        recursive   (bool[False])
    """
    folder = [folder] if isinstance (folder, str) else folder
    folder = [osp.abspath(osp.expanduser(f)) for f in folder]
    ext = [ext] if isinstance (ext, str) else ext
    cond = lambda x, ext: True if ext is None else osp.splitext(x)[-1].lower() in ext

    out = []
    for fold in folder:
        if not recursive:
            out += [f.path for f in os.scandir(fold) if f.is_file() and cond(f.name, ext)]
        else:
            for root, _, files in os.walk(fold):
                out += [osp.join(root, name) for name in files if cond(name, ext)]
    return sorted(out)

def verify_image(name: str, verbose: bool=False) -> bool:
    """
    True:  48 us
    False: 68 us
    jpg and png headers with open('rb') is ~3x to 6x faster, but this is practical and tested
    """
    try:
        im = Image.open(name)
        return True
    except:
        if verbose:
            print(f" Not an Image: {name}")
        return False

def verify_images(images: list, verbose: bool=False) -> list:
    """ check that image list can be opened with PIL"""
    return [im for im in images if verify_image(im, verbose)]

def get_images(folder: Union[str, list, tuple]=".", recursive: bool=False,
               verify: bool=True, verbose: bool=True) -> list:
    """ conditional image file getter
    Args
        folder      (str|list ['.'])  folder | list of folders
        recursive   (bool [False])
        verify      (bool [True]), leverages PIL to read the header of each file
                        may be a bit slower
                        loading verify_image2() should be faster but is not fully tested
        verbose     (default [True])
    """
    _images = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    out = get_files(folder=folder, ext=_images, recursive=recursive)

    if verify:
        out = verify_images(out, verbose)

    if verbose:
        print(f"get_images()-> {len(out)} found")
    return out

def rndlist(inputs: Union[list, tuple, Union[Vector]], num: int=1) -> Any:
    """ returns random subset from list
    Args
        inputs   (iterable)
        num      (int [1]) number of elements returned
    """
    choice = np.random.randint(0, len(inputs), num)
    if isinstance(inputs, Vector):
        return inputs[choice]
    return [inputs[c] for c in choice]


#
# hashing utils
#
def hash_folder(files: Union[str,list], metadata: Collection=None,
                metakey: str="metadata", save: str=None, update: dict=None) -> dict:
    """
        Args
            files       folder name or file list
            [metadata]  iterable same len as files
            [metakey]   key indexing metadata
            [save]      filename to torch.save() or pickle
            [update]    other dict to merge with hashed folder
    Examples
    >>> features = hash_folder(files, m.features, "features", "mydir_features.pt")
    >>> features = torch.load("mydir_features.pt")
    """
    if isinstance(files, str) and osp.isdir(files):
        files = [f.path for f in os.scandir(files) if f.is_file()]
    if metadata is not None:
        assert len(metadata) == len(files)

    out = {}
    _meta = None
    for i, name in enumerate(files):
        if metadata is not None:
            _meta = metadata[i]
        out[name] = hash_file(name, splitname=True, metadata=_meta, metakey=metakey)

    if update is not None:
        out.update(update)

    if isinstance(save, str):
        save = save if osp.splitext(save)[-1] else save + ".pt"
        if WITH_TORCH:
            torch.save(out, save)
        else:
            with open(save, "wb") as _fi:
                pickle.dump(out, _fi)
    return out

def hash_file(filename: str, splitname: bool=False,
              metadata: Any=None, metakey: str="metadata") -> dict:
    """
    hash dictionary
        name:       filename [if splitname: fullname]
        [folder]:   basedir if splitname
        datesize    md5 hash of tuple(mtime, size)
        content     md5 hash of file content
        <metakey>   any extra input

    """
    out = {}
    _st = os.stat(filename)
    if not splitname:
        out['name'] = filename
    else:
        out['folder'], out['name'] = osp.split(filename)
    out['datesize'] = hashlib.md5(json.dumps((_st.st_mtime,_st.st_size), sort_keys=True
                                  ).encode('utf-8')).hexdigest()
    with open(filename, 'rb') as _fi:
        out['content'] = hashlib.md5(_fi.read()).hexdigest()

    if metadata is not None:
        out[metakey] = metadata
    return out

# def check_file(filename, hashdic):
#     pass

def reversedict(dic: str, subkey:str, sort: bool=False) -> list:
    """ returns (subkeyvalue, key)
    given {key_0:{<subkey>:keyval0_i, ..., subkey_n:keyval0_n},
           ...,
           key_m:{<subkey>:keyvalm_i}, ....m}

    returns
        [(keyval0_i, key_0), ..., (keyvalm_1, key_m)]

    works on dict on which each value is a dict.
    assumes every item has same keys
    Exmaple
    >>> x = reversedict(dic, "content")

    """
    _subkeys = list(dic.keys())[0]
    assert subkey in dic[_subkeys].keys(), f"subkey '{subkey}' not found in {_subkeys}"

    out = [(dic[key][subkey], key) for key in dic]
    if sort:
        out = sorted(out, key=lambda x: x[0])
    return out

def get_keys_with_subkeyval(dic: dict, subkey: str, subval: str) -> dict:
    """ return dic subset with subkey:subval
    Example
    >> dic = torch.load("mydir.pt")
    >> get_keys_with_subkeyval(my_dic, "datesize", "c828d7d9d3aafd0b70127aae84208d97")
    """
    return {key:dic[key] for key in dic if dic[key][subkey] == subval}
