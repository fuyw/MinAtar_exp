for i in 1 2 3 4
do 
    for env in "breakout" "freeway"
    do
        python run_offline.py --env_name=$env --seed=$i
    done
done