#!/bin/bash

debug_str="import pydevd;pydevd.settrace('localhost', port=8081, stdoutToServer=True, stderrToServer=True)"
# pydevd module path
export PYTHONPATH=/home/shhs/Desktop/user/soft/pycharm-2018.1.4/debug-eggs/pycharm-debug-py3k.egg_FILES

insert_debug_string()
{
    file=$1
    line=$2
    debug_string=$3
    debug=$4

    value=`sed -n ${line}p $file`
    if [ "$value" != "$debug_str" ] && [ "$debug" = debug ]
    then
    echo "++Insert $debug_string in line_${line}++"
    sed -i "${line}i $debug_str" $file
    fi
}

delete_debug_string()
{
    file=$1
    line=$2
    debug_string=$3

    value=`sed -n ${line}p $file`
    if [ "$value" = "$debug_str" ]
    then
    echo "--Delete $debug_string in line_${line}--"
    sed -i "${line}d" $file
    fi
}


if [ $1 = notebooks/MDN-1D-sine-prediction.ipynb ]
then
    # ./run.sh notebooks/MDN-1D-sine-prediction.ipynb debug
    # tf1.9, python3.6, keras2.2.2
    source $HOME/anaconda3/bin/activate tf_1_9
    export LD_LIBRARY_PATH=/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

    debug=$2
    if [ $debug = debug ]
    then
        cd notebooks
        file="MDN-1D-sine-prediction.py"
        line=16
        insert_debug_string $file $line "$debug_str" $debug
        python $file
        delete_debug_string $file $line "$debug_str"
    else
        jupyter notebook --browser google-chrome
    fi

fi
