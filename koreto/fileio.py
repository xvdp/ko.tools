""" @xvdp fileio
methods to deal with files

    get_files(folders, ext, recursive) # recursive find file
    get_images(folders, recursive)

    hash_file() # hash file, datasizie, content and optional metadata
    hash_folder() # folder or files
    reversedict() # key with subkey

"""
from typing import Union, Any, Collection, Optional
import os
import os.path as osp
import json
import pickle
import hashlib
import numpy as np
from PIL import Image, ImageGrab


from koreto import WITH_TORCH
if WITH_TORCH:
    import torch
    Vector = Union[np.ndarray, torch.Tensor]
else:
    Vector = np.ndarray

def get_last(number: Optional[int] = 1,
             folder: str = '.',
             also_folders: bool = False,
             ext: str = None) -> list:
    """ get last files in folder
    Args
        number  (int [1])  number of files, None: all
        folder  (str ['.'])
        also_folders (bool [False]) return folders
        ext     (str [None]) filter by extension 
    """
    folder = osp.abspath(osp.expanduser(folder))
    files = [f.path for f in os.scandir(folder)
             if (f.is_file() or also_folders) and
             (ext is None or f.name.lower().endswith(ext.lower()))]
    return sorted(files, key=os.path.getmtime, reverse=True)[:number]


def clip_image(name="image.png"):
    """Saves image from clipboard
    """
    name = osp.abspath(osp.expanduser(name))
    _, ext = osp.splitext(name)
    if ext.lower() not in (".png", ".jpg"):
        name += ".png"
    image = ImageGrab.grabclipboard()
    if image:
        image.save(name)
        print(f"Saved clipboard {image.size} to '{name}'")
    else:
        print("No image found in the clipboard.")


def get_files(folder: Union[str, list, tuple] = ".",
              ext: Union[str, list, tuple] = None,
              recursive: bool = False,
              filter_text: str = '',
              sortkey: Optional[str] = None) -> list:
    """ conditional file getter

    Args
        folder      (str|list ['.'])  folder | list of folders
        ext         (str|list [None]) file extensions, default, any
        recursive   (bool [False])
        filter_text (str '') filters file paths containing <filter_ext>
        sortkey     (str [None]) None: alphabetically | 'mtime', 'ctime', 'atime', 'size'

    Examples:
    # return all .flac and .wav files sorted by modification time in current folder
    >>> get_files(ext=('.flac', '.wav'), sortkey='mtime')
    # return all .png files children to home folder recursively sorted by size
    >>> get_files(folder='~', ext='.png', sortkey='size')
    """
    folder = [folder] if isinstance (folder, str) else folder
    folder = [osp.abspath(osp.expanduser(f)) for f in folder]
    ext = [ext] if isinstance (ext, str) else ext
    cond = lambda x, ext: True if ext is None else osp.splitext(x)[-1].lower() in ext
    out = []
    for fold in folder:
        if not recursive:
            out += [f.path for f in os.scandir(fold) if f.is_file() and
                    cond(f.name, ext) and filter_text in f.path]
        else:
            for root, _, files in os.walk(fold):
                out += [osp.join(root, name) for name in files if cond(name, ext)
                        and filter_text in osp.join(root, name)]
    key = {} if sortkey is None else {'key':osp.__dict__[f'get{sortkey}']}
    return sorted(out, **key)


def verify_image(name: str, verbose: bool = False) -> bool:
    """
    True:  48 us
    False: 68 us
    jpg and png headers with open('rb') is ~3x to 6x faster, but this is practical and tested
    """
    try:
        Image.open(name)
        return True
    except:
        if verbose:
            print(f" Not an Image: {name}")
        return False


def verify_images(images: list, verbose: bool = False) -> list:
    """ check that image list can be opened with PIL"""
    return [im for im in images if verify_image(im, verbose)]


def get_images(folder: Union[str, list, tuple] = ".",
               recursive: bool = False,
               verify: bool = True,
               sortkey: Optional[str] = None,
               verbose: bool = True) -> list:
    """ conditional image file getter
    Args
        folder      (str|list ['.'])  folder | list of folders
        recursive   (bool [False])
        verify      (bool [True]), leverages PIL to read the header of each file
                        may be a bit slower
                        loading verify_image2() should be faster but is not fully tested
        sortkey     (str [None]) None: alphabetically | 'mtime', 'ctime', 'atime', 'size'
        verbose     (default [True])
    """
    _images = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    out = get_files(folder=folder, ext=_images, recursive=recursive, sortkey=sortkey)

    if verify:
        out = verify_images(out, verbose)

    if verbose:
        print(f"get_images()-> {len(out)} found")
    return out

#
# hashing utils
#
def hash_folder(files: Union[str, list],
                metadata: Optional[Collection] = None,
                metakey: str = "metadata",
                save: str = None,
                update: dict = None) -> dict:
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


def hash_file(filename: str,
              splitname: bool = False,
              metadata: Any = None,
              metakey: str = "metadata") -> dict:
    """
    returns a dictionary with md5 hash of date and content 
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


def reversedict(dic: str, subkey: str, sort: bool = False) -> list:
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
