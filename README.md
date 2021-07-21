# kotools
repository to collect some common tools i use to manage learning


### logging
`class Plog`         running log to collect to pandas csv <br>
`class Col`          color print codes<br>

### memory
`class GPUse`        thin wrap around nvidia-smi <br>
`class CPUse`        thin wrap around psutil.virtualmemory <br>
`class ObjTrace`     GPU and CPU collection based on ObjDict <br>

### general
`class ObjDict`       thin wrap on dictionary for accessing keys as objects, read and write to yaml and json <br>
`sround(x, digits=1)` smart round, to highest digits <br>
`deepclone(x)`        similar to deepcopy, converting torch tensors to cpu

### training 
`class Schedule`    scheduler, linear, exponential, with noisy periodic functions
