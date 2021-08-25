## setup jemalloc
`export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"`

`export LD_PRELOAD=${your_python_env}/lib/libjemalloc.so`

## test_pytorch1.9

`python acc.py`

## test_ipex1.8

`python acc_ipex.py`

## install ipex
`python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable`
