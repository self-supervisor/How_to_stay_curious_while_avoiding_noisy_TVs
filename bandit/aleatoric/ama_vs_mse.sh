reward_functions=(ama)
repeats=(1 2 3)
update_steps=100000
plot_frequency=1000
for repeat in ${repeats[@]}; do
    for reward_function in ${reward_functions[@]}; do 
        rm -r data/*png
        python train_bandit.py --reward_function $reward_function --plot_frequency $plot_frequency --update_steps $update_steps
    done
done
#python analyse_results.py
