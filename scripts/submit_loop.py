#!/bin/bash

base_model='t5-large'

for i in 1 2 3; do
    for formatting in '{A}' '{O}' '{S}' '{A} {S} {O}' 'empty'; do
        # formatting the formatting variable to make more readable in model names
        fmt_name=$(echo $formatting | sed 's/[{}]//g')
        fmt_name=${fmt_name// /}

        # main qsub command to send model to grid
        echo run.sh --transformer $base_model --bsz 4 --rand-seed $i --formatting '"'$formatting'"' --path models/$base_model-$fmt_name/seed-$i;
        qsub -cwd -j yes -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='volta' -l osrel='*' -o LOGs/$base_model-$fmt_name-seed-$i run.sh --transformer $base_model --bsz 4 --rand-seed $i --formatting '"'$formatting'"' --path models/$base_model-$fmt_name/seed-$i;
    done
done
