#!/bin/bash
# Component analysis under the long-tailed setting (imbalance ratio 10), Tables 1-2.
# Usage: bash scripts/run_cub.sh [config ...]   (default: all configurations)
set -e
cd "$(dirname "$0")/.."

PARTS="--use-parts --use-momentum-teacher"
DUAL="--enable-pseudo-labeling --enable-novel-pseudo"
NO_TAIL="--ablate-adaptive-capacity"

flags() {
  case "$1" in
    baseline)     echo "" ;;
    dual)         echo "$DUAL" ;;
    proto_notail) echo "$PARTS $NO_TAIL" ;;
    proto)        echo "$PARTS" ;;
    full_notail)  echo "$PARTS $DUAL $NO_TAIL" ;;
    full)         echo "$PARTS $DUAL" ;;
    *) echo "unknown config: $1" >&2; exit 1 ;;
  esac
}

for cfg in ${@:-baseline dual proto_notail proto full_notail full}; do
  python train.py --dataset-name cub200 --exp-name cub_${cfg} $(flags "$cfg")
done
