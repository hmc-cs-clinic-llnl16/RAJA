#!/bin/bash

rm -rf build-pathcc-release 2>/dev/null
mkdir build-pathcc-release && cd build-pathcc-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/pathcc.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-pathcc-release \
  "$@" \
  ${RAJA_DIR}
