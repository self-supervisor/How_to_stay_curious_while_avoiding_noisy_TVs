# pix2pix experiment examples
seeds=(1 2 3)
amas=(true false)
noisy_tvs=(true false)

for noisy_tv in ${noisy_tvs[@]}; do 
    for ama in ${amas[@]}; do
        for seed in ${seeds[@]}; do 
            mpiexec -n 1 --allow-run-as-root python run.py --noisy_tv $noisy_tv --env mario --env_kind mario --feat_learning pix2pix --ama $ama --seed $seed
        done
    done
done

# idf experiment examples
seeds=(1 2 3)
amas=(true false)
noisy_tvs=(true false)

for noisy_tv in ${noisy_tvs[@]}; do 
    for ama in ${amas[@]}; do
        for seed in ${seeds[@]}; do 
            mpiexec -n 1 --allow-run-as-root python run.py --noisy_tv $noisy_tv --env BankHeistNoFrameskip-v4 --env_kind atari --feat_learning idf --ama $ama --seed $seed
        done
    done
done
