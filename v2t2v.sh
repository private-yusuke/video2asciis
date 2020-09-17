#!/bin/sh
size=12

if [ $# -lt 3 ]; then
    echo "Usage: ./v2t2v.sh <input> <output> <font>" >&2
    exit 1
fi

python3 v2t.py "$1" -s "$size" tmp.txt
python3 t2v.py tmp.txt -A "$1" -f "$3" "$2"
rm tmp.txt "$2_tmp.mp4"
