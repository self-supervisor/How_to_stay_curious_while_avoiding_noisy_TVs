rm *npy
rm *png
rm -r storage/*
rm *mp4
frames_before_resets=(8000000000)
environment=MiniGrid-MultiRoom-N6-v0
randomise_env=False
frames=5000000
random_seeds=(95 96 97 98 99)

for frames_before_reset in ${frames_before_resets[@]}; do

    
    random_action=True
    reward_weighting=10
    noisy_tv=(True False)
    curiosity=(True)
    uncertainty=(True)
    save_interval=2000
    normalise_rewards=True #False
    icm_lr=0.001
    visualizing=False
   
    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --random_action True --visualizing $visualizing --normalise_rewards True --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_random_action_${random_action}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait
   
    random_action=False
    reward_weighting=10
    noisy_tv=(True False)
    curiosity=(True)
    uncertainty=(True)
    save_interval=2000
    normalise_rewards=True
    icm_lr=0.0001
    visualizing=False
   
    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --random_action False --visualizing $visualizing --normalise_rewards True --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait

    reward_weighting=10
    icm_lr=0.001
    noisy_tv=(True False)
    curiosity=(True)
    uncertainty=(False)
    normalise_reward=True

    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --visualizing $visualizing --random_action False --normalise_rewards $normalise_reward --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --icm_lr $icm_lr --reward_weighting $reward_weighting --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait

    reward_weighting=10
    icm_lr=0.0001
    noisy_tv=(True False)
    curiosity=(False)
    uncertainty=(False)
    normalise_reward=True

    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --visualizing $visualizing --random_action False --normalise_rewards $normalise_reward --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget} --icm_lr $icm_lr --reward_weighting $reward_weighting --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --frames_before_reset $frames_before_reset & 
               done
           done
      done
    done
    wait
    python3 -m scripts.plot --environment $environment
done
