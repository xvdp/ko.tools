# kotools
repository to collect some common tools i use to manage learning
generally pytorch


### general
`ObjDict()`         class, thin wrap on Dict for accessing keys as objects, read and write to yaml and json <br>
`sround()`          smart round, to highest digits <br>

`deepclone()`       similar to deepcopy, clone and detach torch tensors to cpu <br>

### logging
`Plog()`            class, running log to collect to pandas csv <br>
`Col()`             class, color print codes<br>

### memory
`GPUse()`           class, thin wrap around nvidia-smi <br>
`CPUse()`           class, thin wrap around psutil.virtualmemory <br>
`ObjTrace()`        class, GPU and CPU collection based on ObjDict <br>

### training 
`Schedule()`        scheduler class, linear, exponential, with noisy periodic functions <br>

### grids
`mgrid()`           fast n dim meshgrid with layout and column order options; as np `np_mgrid()` <br>
`mgrid_pos()`       grid indices; as np `np_mgrid_pos()` <br>

### random sampling
`unique_randint()`  non repeating random ints (torch | numpy)<br>

### cameras
*Camera functions only built for pytorch* <br>
`pixels_to_rays()`  pixels to rays given camera intrinsics (torch)<br>
`points_to_pixels()`pixels to rays given camera intrinsics (torch)<br>
`rotate_rays()`     rotate rays by transform (torch)<br>
`Camera()`          camera class, io, intrinsics, extrinsics (torch)
