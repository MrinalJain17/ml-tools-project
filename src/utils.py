from tqdm.auto import tqdm
import os
import numpy as np
import pandas as pd
random_seed=23

# rounds are simulated from the full dataset
def simulate_rounds(model, rewards, actions_hist, ttime, X_global, y_global, batch_st, batch_end, fit=True):
    start = pd.Timestamp.now()
    
    np.random.seed(batch_st)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(random_seed)
    if fit:
        model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist],
                  warm_start = True)
    
    ttime.append((pd.Timestamp.now() - start)/pd.Timedelta('1m'))
    
    return new_actions_hist


def run_simulation(X, y, model_dict, reward_dict, ttime_dict, 
                   min_batch, incr_batch_pct):
    # create batch index
    n = 0
    incrs = list()
    ns = [0]
    nchoices = y.shape[1]
     
    while n<len(X):
        incr = max(min_batch, int(incr_batch_pct*n))
        if n+incr>len(X):
            incr = len(X)-n
        incrs.append(incr)
        n+=incr
        ns.append(n)

    print(f'The models will be refit {len(ns)-2} times')

    # initial seed - all policies start with the same small random selection of actions/rewards
    first_batch = X[:min_batch, :]

    action_chosen = np.random.randint(nchoices, size=min_batch)
    rewards_received = y[np.arange(min_batch), action_chosen]

    # fitting models for the first time
    for k, model in model_dict.items():
        model.fit(X=first_batch, a=action_chosen, r=rewards_received)

    # these lists will keep track of which actions does each policy choose
    action_dict = {z:action_chosen.copy() for z in model_dict.keys()}

    # now running all the simulation
    for i in tqdm(range(2, len(ns)), leave=False):
        batch_st = ns[i-1]
        batch_end = ns[i]

        for k in model_dict:
            action_dict[k] = simulate_rounds(model_dict[k],
                                                 reward_dict[k],
                                                 action_dict[k],
                                                 ttime_dict[k],
                                                 X, y,
                                                 batch_st, batch_end)

    return model_dict, reward_dict, action_dict, ttime_dict, ns[2:]
