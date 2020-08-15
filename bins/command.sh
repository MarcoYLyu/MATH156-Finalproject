#!/bin/sh

# Author: Yi Lyu
# Copyright (c) marcoylyu.github.io

SCRIPT_DIR=`dirname $0`
# get root directory
temp=$pwd
cd $SCRIPT_DIR
cd ..
ROOT_DIR="$(pwd)"
cd $temp

FILENAMES=('CMakeFiles/' 'CMakeCache.txt' 'Makefile' 'build/' 'cmake_install.cmake')
ALGOLIB='algorithms.cpython-36m-darwin.so'
DATANAMES=('.data.pickle' '.ann/' '.math156.db' '.models.pickle' 'headers.csv')

SCRIPTINFO="RUN_MODEL 0.1.0"
USAGE="Usage: $0 [arguments]"
ARGUMENTS=('compile' 'run' 'run-model' 'clean' 'clean-data' '--help')
DESCRIPTIONS=('compile the C++ library for the models'
              'compile and run the code for prediction models'
              'run the code for prediction models'
              'delete the files created by Cmake'
              'clean the database and models produced'
              'receive help')

len=${#ARGUMENTS[@]}

compile () {
    cd "${ROOT_DIR}/src"
    rm -rf $ALGOLIB
    cmake .
    make
}

run () {
    compile
    python3 main.py
}

run_model () {
    cd "${ROOT_DIR}/src"
    python3 main.py
}

clean () {
    cd "${ROOT_DIR}/src"
    for filename in "${FILENAMES[@]}";
    do
        rm -rf $filename
    done
    echo "-- Cleaning Cmake files - done"
}

clean_data () {
    cd "${ROOT_DIR}/data"
    for datafile in "${DATANAMES[@]}";
    do
        rm -rf $datafile
    done
    echo "-- Cleaning data model - done"
}

case $1 in 
    "compile")
        compile
        ;;
    "run")
        run
        ;;
    "clean")
        clean
        ;;
    "clean-data")
        clean_data
        ;;
    "clean-all")
        clean
        clean_data
        ;;
    "run-model")
        run_model
        ;;
    "--help")
        echo $SCRIPTINFO
        echo " "
        echo $USAGE
        echo " "
        set -f
        for ((i = 0; i < $len; ++i));
        do
            printf "%-10s \t %-20s" "${ARGUMENTS[i]}" "${DESCRIPTIONS[i]}"
            printf "\n"
        done
        ;;
    *)
        echo "Unknown option argument: $1"
        echo "More info: $0 --help"
        ;;
esac