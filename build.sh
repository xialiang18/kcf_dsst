#!/usr/bin/env bash
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build && cd build

#cmake   -DPRECISION=ON \
cmake   -DRESULT_SAVE_PHOTO=ON \
        ..
#        -DPERFORMANCE=ON \
#        ..

#cmake ..
make -j4

mv track ..