from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch

from src.Environment.Environment import *


# Test DQN
def test(args, T, dqn, val_mem, metrics, results_dir, env_args, evaluate=False):
    env_args['random_coverage'] = False
    env_args['random_size'] = False
    env_args['dataset_path'] = 'empty'
    env = Environment(EnvironmentParams(env_args))
    metrics['steps'].append(T)
    T_rewards, T_Qs, T_overlap, T_time_save = [], [], [], []
    # Test performance over several episodes
    done = True
    truncated = False
    for _ in range(args.evaluation_episodes):
        while True:
            if done or truncated:
                state, info = env.reset(False)
                reward_sum, done, truncated = 0, False, False

            
            action = dqn.act_e_greedy(state[0], state[1],state[2],state[3])  # Choose an action ε-greedily
            if any(info):
                    action = dqn.act(state[0], state[1], state[2], state[3])
                    ac = env.get_heuristic_action(info)
                    for i, a in enumerate(ac):
                        if a is not None:
                            action[i] = a
            state, reward, done, truncated, info = env.step(action)  # Step

            reward_sum += sum(reward)
            if args.render:
                env.render()
            if done or truncated:
                T_rewards.append(env.rewards.get_cumulative_reward())
                T_overlap.append(env.rewards.get_overlap())
                T_time_save.append(env.rewards.get_time_save())
                break

    # env.close()

    # Test Q-values over validation memory
    for state, battery, last_action, oob in val_mem:  # Iterate over valid states
        T_Qs.append(dqn.evaluate_q(state, battery,last_action,oob))

    avg_reward, avg_Q, avg_overlap, avg_time_save = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs), sum(T_overlap) / len(
        T_overlap), sum(T_time_save)/len(T_time_save)
    if not evaluate:
        # Save model parameters if improved
        if avg_reward > metrics['best_avg_reward']:
            metrics['best_avg_reward'] = avg_reward
            dqn.save(results_dir)

        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        metrics['Qs'].append(T_Qs)
        metrics['overlap'].append(T_overlap)
        metrics['time_save'].append(T_time_save)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot
        _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
        _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)
        _plot_line(metrics['steps'], metrics['overlap'], 'Overlap', path=results_dir)
        _plot_line(metrics['steps'], metrics['time_save'], 'Time Save', path=results_dir)

    # Return average reward and Q-value
    return avg_reward, avg_Q, avg_overlap, avg_time_save


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
        1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour),
                         name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent),
                          name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)