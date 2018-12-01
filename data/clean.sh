#!/bin/bash
echo

let chunksize=100
filecount=$(find profiles_11-19/* -type f | wc -l)

echo ========== RUNNING DATA_CLEANER.PY
echo ========== chunks of $chunksize for $filecount files 

let start=0
let end=start+chunksize

while [ $end -le $filecount ]; do
    if [ $start -eq $filecount ]; then
        break
    fi 

    echo \[$start --\> $end\]
    python data_cleaner.py profiles_11-19/ $start $end

    let start=end
    let end=start+chunksize

    if [ $end -gt $filecount ]; then
        let end=filecount
    fi

    echo sleep for a bit to let things cooldown
    sleep 2m
done

echo ========== COMPLETED RUNNING FOR FIRST DATASET
echo

echo ========== RUNNING DATA_CLEANERNEW.PY 
echo ========== there are only 2 files to run so running all

python data_cleanerNEW.py profiles_11-29/

echo ========== COMPLETED RUNNING FOR SECOND DATASET
echo

echo ========== COMPILING CREATED JSON FILES

python _data.py

echo ========== DONE 

