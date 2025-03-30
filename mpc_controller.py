from env import MultipleHPCooling
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize
import os
import sys

# Fix paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)

# Load config
with open('sim_config/config0.json', 'r') as f:
    config = json.load(f)

config['control_type'] = 'mpc'
test_number = config['test_number']

# Create MPC result directory
path_results = 'data/result/MPC'
if not os.path.exists(path_results):
    os.makedirs(path_results)


class MPController:
    def __init__(self, env, prediction_horizon):
        self.env = env
        self.prediction_horizon = prediction_horizon
        self.n_buildings = len(config['building_list'])
        self.simulation_end = config['simulation_start'] + (config['n_days'] * 24 * 3600) - 2 * config[
            'simulation_step']

        # Store building parameters
        self.building_params = {}
        for building_id in config['building_list']:
            self.building_params[building_id] = {
                'swt_range': config['init_params'][building_id]['swt_range'],
                'm_1_input': config['init_params'][building_id]['m_1_input'],
                'cop_curve': config['init_params'][building_id]['cop_curve_name']
            }

        # Initialize metrics tracking
        self.metrics = defaultdict(list)

    def optimize(self, state, current_time):
        if current_time + config['simulation_step'] >= self.simulation_end:
            return None

        x0 = np.ones(self.n_buildings) * 0.5
        bounds = [(0, 1) for _ in range(self.n_buildings)]

        try:
            result = minimize(self.objective_function,
                              x0,
                              args=(state,),
                              bounds=bounds,
                              method='SLSQP',
                              options={'maxiter': 100})

            if result.success:
                return result.x
            else:
                return x0
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return x0

    def objective_function(self, actions, state):
        try:
            weather_pred = state[self.n_buildings * 9 + self.n_buildings * (self.prediction_horizon + 1)
                                 :self.n_buildings * 9 + self.n_buildings * (
                        self.prediction_horizon + 1) + self.prediction_horizon + 1]
            price_pred = state[self.n_buildings * 9 + self.n_buildings * (self.prediction_horizon + 1) + (
                        self.prediction_horizon + 1)
                               :self.n_buildings * 9 + self.n_buildings * (self.prediction_horizon + 1) + 2 * (
                        self.prediction_horizon + 1)]
            pv_pred = state[self.n_buildings * 9 + self.n_buildings * (self.prediction_horizon + 1) + 2 * (
                        self.prediction_horizon + 1)
                            :self.n_buildings * 9 + self.n_buildings * (self.prediction_horizon + 1) + 3 * (
                        self.prediction_horizon + 1)]

            control_signals = []
            for i, action in enumerate(actions):
                building_id = config['building_list'][i]
                min_temp = self.building_params[building_id]['swt_range'][0]
                max_temp = self.building_params[building_id]['swt_range'][1]
                mass_flow_rate = self.building_params[building_id]['m_1_input']

                if pv_pred[0] > 0:
                    if action < 0.3:
                        action = 0.5
                    swt = min_temp + (max_temp - min_temp) * (abs((action - 0.3) - 0.7) / 0.7)
                    control_signals.append([mass_flow_rate, float(swt)])
                else:
                    if action < 0.3:
                        control_signals.append([0.0, max_temp])
                    else:
                        swt = min_temp + (max_temp - min_temp) * (abs((action - 0.3) - 0.7) / 0.7)
                        control_signals.append([mass_flow_rate, float(swt)])

            next_state, reward, done, _ = self.env.step(control_signals)

            electricity_cost = -reward

            w1 = 1.0  # weight for electricity cost
            w2 = 0.5  # weight for self-consumption

            if hasattr(self.env, 'result_df') and len(self.env.result_df) > 0:
                last_row = self.env.result_df.iloc[-1]
                pv_to_hp = last_row['pv_to_hp']
                total_ee_hp = last_row['total_ee_hp']

                if total_ee_hp > 0:
                    self_consumption_penalty = w2 * (1 - pv_to_hp / total_ee_hp) if pv_pred[0] > 0 else 0
                else:
                    self_consumption_penalty = 0
            else:
                self_consumption_penalty = 0

            total_cost = w1 * electricity_cost + self_consumption_penalty
            return total_cost

        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            return 1e6

    def log_metrics(self, control_signals, reward):
        self.metrics['rewards'].append(reward)
        self.metrics['costs'].append(-reward)

    def plot_training_curves(self):
        plt.figure(figsize=(10, 5))

        # Policy Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['costs'], label='Policy Loss')
        plt.title('Policy Loss')
        plt.xlabel('Steps')
        plt.ylabel('Cost')

        # Reward Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['rewards'], label='Reward Loss')
        plt.title('Training Metrics\nReward Loss')
        plt.xlabel('Steps')
        plt.ylabel('Reward')

        plt.tight_layout()
        plt.savefig(os.path.join(path_results, f'training_curves_{test_number}.png'))
        plt.close()


# Create environment and controller
env = MultipleHPCooling(config)
mpc_controller = MPController(env, config['prediction_horizon'])

# Main control loop
observation = env.reset()
done = False
current_time = config['simulation_start']
last_printed_day = -1

print(f"\nStarting MPC simulation:")
print(f"Start time: {current_time}")
print(f"End time: {mpc_controller.simulation_end}")
print(f"Duration: {config['n_days']} days")
print(f"Simulation step: {config['simulation_step']} seconds\n")

while not done and current_time < mpc_controller.simulation_end:
    try:
        current_day = (current_time - config['simulation_start']) // 86400

        if current_day != last_printed_day:
            print(f"Day: {current_day:02d}, Cost: {env.episode_cost:.2f}, Reward: {env.episode_reward:.2f}")
            last_printed_day = current_day
            mpc_controller.plot_training_curves()

        action = mpc_controller.optimize(observation, current_time)
        if action is None:
            print(f"\nReached simulation end time: {current_time}")
            break

        control_signals = []
        for i, act in enumerate(action):
            building_id = config['building_list'][i]
            min_temp = config['init_params'][building_id]['swt_range'][0]
            max_temp = config['init_params'][building_id]['swt_range'][1]
            mass_flow_rate = config['init_params'][building_id]['m_1_input']

            if act < 0.3:
                control_signals.append([0.0, max_temp])
            else:
                swt = min_temp + (max_temp - min_temp) * (abs((act - 0.3) - 0.7) / 0.7)
                control_signals.append([mass_flow_rate, float(swt)])

        observation, reward, done, info = env.step(control_signals)
        mpc_controller.log_metrics(control_signals, reward)
        current_time += config['simulation_step']

    except Exception as e:
        print(f"\nSimulation stopped with error at time {current_time}")
        print(f"Error details: {str(e)}")
        break

# Plot final curves
mpc_controller.plot_training_curves()

print("\nFinal Results:")
print(f"Total Cost: {env.episode_cost:.2f}")
print(f"Self Sufficiency: {env.self_sufficiency:.2f}%")
print(f"Self Consumption: {env.self_consumption:.2f}%")

# Save results
hyperP = pd.DataFrame(
    [[test_number, config['prediction_horizon'], env.episode_cost,
      env.self_sufficiency, env.self_consumption]],
    columns=['Test number', 'Prediction horizon', 'Electricity cost',
             'Self sufficiency', 'Self consumption'])

runsLogName = "data/result/summary_mpc_trials.csv"

if not os.path.exists(runsLogName):
    hyperP.to_csv(path_or_buf=runsLogName,
                  sep=',', decimal='.', index=False)
else:
    runsLog = pd.read_csv(runsLogName, sep=',', decimal='.')
    runsLog = pd.concat([runsLog, hyperP])
    runsLog.to_csv(path_or_buf=runsLogName,
                   sep=',', decimal='.', index=False)
