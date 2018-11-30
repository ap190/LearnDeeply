#!/bin/bash
echo

let chunksize=100
filecount=$(find profiles_11-19/* -type f | wc -l)

echo Running data_cleaner.py in chunks of $chunksize for  $filecount files ...

let start=0
let end=start+chunksize

while [ $end -le $filecount ]; do
    echo \[$start --\> $end\]
    python data_cleaner.py profiles_11-19/ $start $end

    let start=end
    let end=start+chunksize

    if [ $end -gt $filecount ]; then
        let end=filecount
    fi

    echo sleep for a bit to let things cooldown
    sleep 1m
done

echo Remember to combine the individual data.json files! 