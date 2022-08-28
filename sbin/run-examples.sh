#!/bin/bash
set -eo

DIR=$(dirname $0)
cd ${DIR}/../

export EXAMPLES_DIR="src/rlplg/examples"
ls $EXAMPLES_DIR | grep "\.py" | xargs -I % python $EXAMPLES_DIR/%
