#for meth in "SGD"
#do
#    for width in 128 256 1024
#    do 
#        for year in 2015 2016 2017 2018 2019
#        do
#            for state in "NY" "CA" "TX" "NC" "WA" "MA" "RI" "ND" "SD" 
#            do
#                python3 folk_train.py --fair "NONE" --meth "NONE" --width $width --year $year --state $state &
#            done
#            wait
#        done
#    done
#done

for fair in "LP" "SENSR" "JOHN" 
do 
    for meth in "FAIR-IBP" "FAIR-PGD"
    do
        for width in 128 256 1024
        do 
            for year in 2015 2016 2017 2018 2019
            do
                for state in "NY" "CA" "TX" "NC" "WA" "MA" "RI" "ND" "SD"
                do
                    python3 folk_train.py --fair $fair --meth $meth --width $width --year $year --state $state  &
                done
                wait
            done
            
        done
    done
done