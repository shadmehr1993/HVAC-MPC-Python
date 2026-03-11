# mpc.py — Robust Grey-Box MPC with Physical Constraints

from env import MultipleHPCooling
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize
import os

# --------------------------------
# Setup paths / config
# --------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)

with open('sim_config/config0.json', 'r') as f:
    config = json.load(f)

config['control_type'] = 'mpc'
test_number = config['test_number']

path_results = 'data/result/MPC'
os.makedirs(path_results, exist_ok=True)


class RobustGreyBoxMPC:
    """
    Robust Grey-Box MPC with physical constraints and smooth control
    """

    def __init__(self, env, prediction_horizon):
        self.env = env
        self.H = prediction_horizon
        self.n_buildings = len(config['building_list'])
        self.dt = config['simulation_step'] / 3600.0  # hours

        self.simulation_end = (
                config['simulation_start'] +
                (config['n_days'] * 24 * 3600) -
                config['simulation_step']
        )

        # Building parameters and COP curves
        self.building_params = {}
        for bid in config['building_list']:
            with open(f"data/cop_curves/{config['init_params'][bid]['cop_curve_name']}.json", "r") as f:
                cop_curve = json.load(f)

            self.building_params[bid] = {
                'swt_range': config['init_params'][bid]['swt_range'],
                'm_1_input': config['init_params'][bid]['m_1_input'],
                'V_tank': config['init_params'][bid]['V_tank'],
                'm_2_max': config['init_params'][bid]['m_2_max'],
                'm_2_nom': config['init_params'][bid]['m_2_nom'],
                'cop_curve': cop_curve,
                'Q_max': config['init_params'][bid]['Q_max_building'] / 1000.0,  # kW
            }

        self.eta_ac_dc = config["converter_efficiency"]
        self.battery_capacity = config["battery_size"]
        self.battery_rte = config["round_trip_efficiency"]

        # Improved thermal model parameters
        self.c_water = 4.186  # kJ/(kg·K)
        self.rho_water = 1000  # kg/m³

        # Tank heat loss coefficient (calibrated for better accuracy)
        self.U_tank = 0.5  # W/(m²·K) - heat loss coefficient
        self.tank_surface_ratio = 6.0  # Surface to volume ratio (m²/m³)

        # Control smoothing parameters
        self.alpha_smooth = 0.7  # Exponential smoothing for control changes
        self.max_delta_swt = 2.0  # K maximum SWT change per step
        self.max_delta_flow = 0.3  # Maximum flow rate change (fraction of nominal)

        # Previous controls for smoothing
        self.prev_controls = None

        # MPC objective weights
        self.w_energy_cost = 1.0
        self.w_comfort = 0.5  # Increased comfort weight
        self.w_battery_deg = 0.05
        self.w_control_smooth = 0.2  # Penalty for control changes
        self.w_tank_penalty = 1.0  # Penalty for tank temperature violations

        # Physical constraints
        self.tank_T_min = 278.15  # K (safe minimum)
        self.tank_T_max = 293.15  # K (safe maximum)
        self.tank_T_target = 285.15  # K (comfortable target)
        self.battery_soc_min = 0.15
        self.battery_soc_max = 0.85

        # Metrics tracking
        self.metrics = defaultdict(list)

        # State indices
        self._idx = self._build_indices()

        # Control mode tracking
        self.hp_on_duration = np.zeros(self.n_buildings)
        self.min_on_time = 2  # Minimum ON duration (steps)
        self.min_off_time = 2  # Minimum OFF duration (steps)

    def _build_indices(self):
        """Build indices to parse observation vector"""
        H = self.H + 1
        nb = self.n_buildings
        sim_vars_len = 9 * nb
        loads_len = nb * H

        idx = {}
        idx['H'] = H
        idx['sim_vars'] = slice(0, sim_vars_len)
        idx['loads'] = slice(sim_vars_len, sim_vars_len + loads_len)

        base = sim_vars_len + loads_len
        idx['weather'] = slice(base, base + H)
        idx['price'] = slice(base + H, base + 2 * H)
        idx['pv'] = slice(base + 2 * H, base + 3 * H)
        idx['bess_soc'] = base + 3 * H
        idx['time'] = slice(base + 3 * H + 1, base + 3 * H + 3)

        return idx

    def _unscale(self, s):
        """Unscale normalized observation"""
        s = np.asarray(s, dtype=float)
        mins = np.asarray(self.env.min_values, dtype=float)
        maxs = np.asarray(self.env.max_values, dtype=float)
        return mins + s * (maxs - mins + 1e-12)

    def _parse_observation(self, obs_scaled):
        """Extract all relevant data from observation"""
        raw = self._unscale(obs_scaled)
        idx = self._idx
        H = idx['H']

        # Current building states
        tank_T = np.zeros(self.n_buildings)
        primary_swt = np.zeros(self.n_buildings)
        primary_rwt = np.zeros(self.n_buildings)
        secondary_flow = np.zeros(self.n_buildings)

        for i in range(self.n_buildings):
            start = i * 9
            tank_T[i] = raw[start + 2]
            primary_swt[i] = raw[start + 0]
            primary_rwt[i] = raw[start + 1]
            secondary_flow[i] = raw[start + 6]

        # Predictions over horizon
        loads = raw[idx['loads']].reshape(self.n_buildings, H)
        weather = raw[idx['weather']]
        prices = raw[idx['price']]
        pv = raw[idx['pv']]
        bess_soc = raw[idx['bess_soc']]

        return {
            'tank_T': tank_T,
            'primary_swt': primary_swt,
            'primary_rwt': primary_rwt,
            'secondary_flow': secondary_flow,
            'loads': loads,
            'weather': weather,
            'prices': prices,
            'pv': pv,
            'bess_soc': bess_soc
        }

    def _calculate_cop(self, cop_curve, t_out_C, swt_K):
        """Calculate COP from polynomial curve with safety bounds"""
        t_out_K = t_out_C + 273.15

        cop = (cop_curve['intercept']
               + cop_curve['coefTout'] * t_out_K
               + cop_curve['coefSWT'] * swt_K
               + cop_curve['coefTout2'] * (t_out_K ** 2)
               + cop_curve['coefSWT2'] * (swt_K ** 2)
               + cop_curve['coefToutSWT'] * t_out_K * swt_K)

        # Safety bounds on COP
        cop = np.clip(cop, 1.5, 10.0)
        return cop

    def _predict_tank_temperature(self, tank_T_prev, Q_hp, Q_load, m_secondary, V_tank, t_ambient):
        """
        Improved grey-box thermal model for tank

        Energy balance with heat losses:
        C_tank * dT/dt = Q_hp - Q_load - Q_loss
        """
        # Tank thermal capacity
        C_tank = self.rho_water * V_tank * self.c_water  # kJ/K

        # Heat loss to ambient (improved model)
        A_tank = self.tank_surface_ratio * V_tank  # m²
        Q_loss = self.U_tank * A_tank * (tank_T_prev - t_ambient) / 1000.0  # kW
        Q_loss = Q_loss * self.dt  # kWh

        # Mixing efficiency (secondary loop extracts heat)
        mixing_factor = min(m_secondary / 2.0, 1.0)  # Normalize

        # Net energy change
        delta_E = (Q_hp - Q_load * mixing_factor - Q_loss) * 3600.0  # kJ

        # Temperature change
        delta_T = delta_E / C_tank
        tank_T_next = tank_T_prev + delta_T

        # Hard constraints
        tank_T_next = np.clip(tank_T_next, self.tank_T_min, self.tank_T_max)

        return tank_T_next

    def _predict_battery_soc(self, soc_prev, P_charge_kW, dt_hours):
        """Predict battery SOC with round-trip efficiency"""
        if P_charge_kW > 0:  # Charging
            energy_in = P_charge_kW * dt_hours * np.sqrt(self.battery_rte)
        else:  # Discharging
            energy_in = P_charge_kW * dt_hours / np.sqrt(self.battery_rte)

        soc_next = soc_prev + energy_in / self.battery_capacity
        soc_next = np.clip(soc_next, self.battery_soc_min, self.battery_soc_max)

        return soc_next

    def _simulate_step(self, state, control, t_idx, data):
        """
        Simulate one step with improved physics
        """
        tank_T = state['tank_T'].copy()
        bess_soc = state['bess_soc']

        hp_energy_total = 0.0
        t_ambient = data['weather'][t_idx] + 273.15  # K

        # For each building
        for i in range(self.n_buildings):
            bid = config['building_list'][i]
            m_flow, swt = control[i]

            params = self.building_params[bid]
            cop_curve = params['cop_curve']
            V_tank = params['V_tank']
            m_nom = params['m_1_input']

            # Thermal load
            Q_load = data['loads'][i, t_idx]  # kWh

            # HP operation
            if m_flow > 0.05 * m_nom:  # HP is ON
                # COP calculation
                cop = self._calculate_cop(cop_curve, data['weather'][t_idx], swt)

                # HP thermal output (based on flow and temperature difference)
                # More conservative estimate
                delta_T = max((tank_T[i] + 8.0) - swt, 1.0)  # Conservative estimate
                Q_hp = self.c_water * m_flow * delta_T * self.dt  # kWh

                # Limit to max capacity
                Q_hp = min(Q_hp, params['Q_max'] * self.dt)

                # Electrical energy
                E_hp = Q_hp / cop
            else:
                Q_hp = 0.0
                E_hp = 0.0

            # Secondary loop flow (load-driven)
            m_secondary = min(Q_load / (self.c_water * 5.0 * self.dt + 1e-6), params['m_2_max'])

            # Update tank temperature
            tank_T[i] = self._predict_tank_temperature(
                tank_T[i], Q_hp, Q_load, m_secondary, V_tank, t_ambient
            )

            hp_energy_total += E_hp

        # Energy management
        pv_available = data['pv'][t_idx] * self.dt  # kWh
        price = data['prices'][t_idx]

        # Simple dispatch
        net_load = hp_energy_total - pv_available

        if net_load > 0:  # Need energy
            battery_available = (bess_soc - self.battery_soc_min) * self.battery_capacity
            battery_discharge = min(net_load / np.sqrt(self.battery_rte),
                                    battery_available * 0.5)  # Conservative discharge

            grid_import = max(net_load - battery_discharge * np.sqrt(self.battery_rte), 0.0)
            battery_charge = -battery_discharge
            grid_export = 0.0
        else:  # Excess PV
            excess = -net_load
            battery_space = (self.battery_soc_max - bess_soc) * self.battery_capacity
            battery_charge = min(excess * np.sqrt(self.battery_rte),
                                 battery_space * 0.5)  # Conservative charge

            grid_export = max(excess - battery_charge / np.sqrt(self.battery_rte), 0.0)
            grid_import = 0.0

        # Update battery
        bess_soc_next = self._predict_battery_soc(bess_soc, battery_charge / self.dt, self.dt)

        # Cost (export credit = 50% of import price)
        cost = price * (grid_import - 0.5 * grid_export)

        next_state = {
            'tank_T': tank_T,
            'bess_soc': bess_soc_next
        }

        return next_state, hp_energy_total, grid_import, cost

    def _control_vector_to_matrix(self, u_flat):
        """Convert flat control vector to (H, N, 2) matrix"""
        # u_flat has shape (H * N * 2)
        # Reshape to (H, N, 2)
        try:
            u_matrix = u_flat.reshape(self.H, self.n_buildings, 2)
        except:
            # Fallback
            u_matrix = np.zeros((self.H, self.n_buildings, 2))
            u_matrix[:, :, 0] = 0.5  # Default to 50% flow
            u_matrix[:, :, 1] = 0.5  # Default to mid SWT
        return u_matrix

    def _matrix_to_physical_controls(self, u_matrix, data):
        """
        Convert normalized [0,1] controls to physical values with smoothing
        """
        controls = np.zeros((self.H, self.n_buildings, 2))

        for i in range(self.n_buildings):
            bid = config['building_list'][i]
            params = self.building_params[bid]

            m_nom = params['m_1_input']
            swt_min, swt_max = params['swt_range']

            for t in range(self.H):
                # Mass flow: binary decision + magnitude
                # If u < 0.3: OFF, else ON with proportional flow
                if u_matrix[t, i, 0] < 0.3:
                    m_flow = 0.0
                else:
                    # Scale from 0.3-1.0 to 0.5-1.0 * m_nom
                    flow_factor = 0.5 + 0.5 * (u_matrix[t, i, 0] - 0.3) / 0.7
                    m_flow = flow_factor * m_nom

                # SWT: interpolate within range, biased toward safe values
                swt_factor = u_matrix[t, i, 1]
                swt = swt_min + swt_factor * (swt_max - swt_min)

                # Apply smoothing from previous controls
                if t == 0 and self.prev_controls is not None:
                    prev_m = self.prev_controls[i, 0]
                    prev_swt = self.prev_controls[i, 1]

                    # Limit changes
                    m_flow = np.clip(m_flow,
                                     prev_m - self.max_delta_flow * m_nom,
                                     prev_m + self.max_delta_flow * m_nom)
                    swt = np.clip(swt,
                                  prev_swt - self.max_delta_swt,
                                  prev_swt + self.max_delta_swt)

                    # Exponential smoothing
                    m_flow = self.alpha_smooth * prev_m + (1 - self.alpha_smooth) * m_flow
                    swt = self.alpha_smooth * prev_swt + (1 - self.alpha_smooth) * swt

                # Ensure physical bounds
                m_flow = max(0.0, min(m_flow, m_nom))
                swt = max(swt_min, min(swt, swt_max))

                controls[t, i, 0] = m_flow
                controls[t, i, 1] = swt

        return controls

    def objective(self, u_flat, initial_state, data):
        """
        MPC objective: minimize cost while maintaining comfort and smooth control
        """
        u_matrix = self._control_vector_to_matrix(u_flat)
        controls = self._matrix_to_physical_controls(u_matrix, data)

        state = initial_state.copy()
        total_cost = 0.0

        for t in range(self.H):
            # Simulate step
            next_state, hp_energy, grid_energy, cost = self._simulate_step(
                state, controls[t], t, data
            )

            # 1. Energy cost
            J_energy = self.w_energy_cost * cost

            # 2. Comfort penalty (squared error from target)
            tank_deviation = np.sum((state['tank_T'] - self.tank_T_target) ** 2)
            J_comfort = self.w_comfort * tank_deviation

            # 3. Hard constraint penalty for tank limits
            J_tank_penalty = 0.0
            for tank_t in state['tank_T']:
                if tank_t < self.tank_T_min:
                    J_tank_penalty += self.w_tank_penalty * ((self.tank_T_min - tank_t) ** 2)
                elif tank_t > self.tank_T_max:
                    J_tank_penalty += self.w_tank_penalty * ((tank_t - self.tank_T_max) ** 2)

            # 4. Battery degradation (cycling)
            if t > 0:
                J_battery = self.w_battery_deg * abs(state['bess_soc'] - next_state['bess_soc'])
            else:
                J_battery = 0.0

            # 5. Control smoothing (penalize rapid changes)
            if t > 0:
                du = np.sum((controls[t] - controls[t - 1]) ** 2)
                J_smooth = self.w_control_smooth * du
            else:
                J_smooth = 0.0

            total_cost += J_energy + J_comfort + J_tank_penalty + J_battery + J_smooth
            state = next_state

        # Terminal cost: prefer final state near target
        J_terminal = 0.2 * np.sum((state['tank_T'] - self.tank_T_target) ** 2)
        total_cost += J_terminal

        return total_cost

    def optimize(self, observation, current_time):
        """
        Solve MPC optimization with robust fallback
        """
        if current_time >= self.simulation_end:
            return None

        # Parse observation
        data = self._parse_observation(observation)

        # Initial state
        initial_state = {
            'tank_T': data['tank_T'].copy(),
            'bess_soc': data['bess_soc']
        }

        # Initial guess: warm start from previous or conservative
        if self.prev_controls is not None:
            # Repeat previous control over horizon
            u0 = np.tile(np.array([0.5, 0.5]), (self.H, self.n_buildings, 1))
            for i in range(self.n_buildings):
                bid = config['building_list'][i]
                m_nom = self.building_params[bid]['m_1_input']
                swt_min, swt_max = self.building_params[bid]['swt_range']

                # Normalize previous controls
                u0[:, i, 0] = self.prev_controls[i, 0] / m_nom
                u0[:, i, 1] = (self.prev_controls[i, 1] - swt_min) / (swt_max - swt_min)
            u0 = u0.flatten()
        else:
            u0 = np.full(self.H * self.n_buildings * 2, 0.5)

        # Bounds: all normalized [0, 1]
        bounds = [(0.0, 1.0)] * len(u0)

        # Solve with multiple attempts
        for attempt in range(2):
            try:
                result = minimize(
                    self.objective,
                    u0,
                    args=(initial_state, data),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-4}
                )

                if result.success or attempt == 1:  # Accept on second attempt
                    u_opt = self._control_vector_to_matrix(result.x)
                    controls_opt = self._matrix_to_physical_controls(u_opt, data)

                    # Store for next iteration
                    self.prev_controls = controls_opt[0].copy()

                    return controls_opt[0]

            except Exception as e:
                if attempt == 0:
                    print(f"Optimization attempt {attempt + 1} failed: {e}")
                    # Try with more conservative initial guess
                    u0 = np.full(self.H * self.n_buildings * 2, 0.4)
                else:
                    print(f"Optimization failed after {attempt + 1} attempts")

        # Ultimate fallback
        return self._get_safe_fallback_control(data, initial_state)

    def _get_safe_fallback_control(self, data, state):
        """Conservative rule-based fallback"""
        controls = np.zeros((self.n_buildings, 2))

        for i in range(self.n_buildings):
            bid = config['building_list'][i]
            params = self.building_params[bid]

            tank_T = state['tank_T'][i]
            load = data['loads'][i, 0]

            # Simple logic: turn on if tank is getting cold or load is high
            if tank_T < self.tank_T_target - 3.0 or load > 10.0:
                controls[i, 0] = 0.7 * params['m_1_input']  # 70% flow
                controls[i, 1] = params['swt_range'][0] + 3.0  # Safe SWT
            else:
                controls[i, 0] = 0.0
                controls[i, 1] = params['swt_range'][0]

        self.prev_controls = controls.copy()
        return controls

    def log_metrics(self, action, reward):
        """Log metrics"""
        self.metrics['rewards'].append(reward)
        self.metrics['costs'].append(-reward)

    def plot_training_curves(self):
        """Plot performance"""
        if len(self.metrics['costs']) == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].plot(self.metrics['costs'], label='Cost', linewidth=0.8)
        axes[0].set_title('Cumulative Cost')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Cost (€)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(self.metrics['rewards'], label='Reward', linewidth=0.8, color='green')
        axes[1].set_title('Reward')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(path_results, f'mpc_performance_{test_number}.png'), dpi=150)
        plt.close()


# --------------------------------
# Main execution
# --------------------------------
def main():
    print("=" * 60)
    print("ROBUST GREY-BOX MPC CONTROLLER")
    print("=" * 60)

    # Create environment
    env = MultipleHPCooling(config, test_mode=True)

    # Create MPC controller
    mpc = RobustGreyBoxMPC(env, config['prediction_horizon'])

    print(f"\nConfiguration:")
    print(f"  Buildings: {config['building_list']}")
    print(f"  Prediction horizon: {config['prediction_horizon']} steps")
    print(f"  Simulation duration: {config['n_days']} days")
    print(f"  Control smoothing: α={mpc.alpha_smooth}")
    print(f"  Max ΔT: {mpc.max_delta_swt} K/step")
    print(f"  Max Δm: {mpc.max_delta_flow * 100:.0f}% /step")
    print()

    # Reset environment
    observation = env.reset()
    done = False
    current_time = config['simulation_start']
    step_count = 0
    last_printed_day = -1

    print("Starting simulation...\n")

    # Main control loop
    while not done and current_time < mpc.simulation_end:
        try:
            current_day = (current_time - config['simulation_start']) // 86400

            if current_day != last_printed_day:
                print(f"Day {current_day:3d} | "
                      f"Cost: {env.episode_cost:8.2f} € | "
                      f"Reward: {env.episode_reward:10.2f}")
                last_printed_day = current_day

                # Plot every 10 days
                if current_day % 10 == 0 and current_day > 0:
                    mpc.plot_training_curves()

            # MPC optimization
            optimal_control = mpc.optimize(observation, current_time)

            if optimal_control is None:
                break

            # Convert to list format
            control_signals = [[float(optimal_control[i, 0]), float(optimal_control[i, 1])]
                               for i in range(mpc.n_buildings)]

            # Execute control
            observation, reward, done, info = env.step(control_signals)

            # Log
            mpc.log_metrics(control_signals, reward)

            current_time += config['simulation_step']
            step_count += 1

        except Exception as e:
            print(f"\nError at time {current_time}: {str(e)}")
            print("Attempting to continue with safe fallback...")

            # Try safe fallback
            try:
                data = mpc._parse_observation(observation)
                initial_state = {'tank_T': data['tank_T'], 'bess_soc': data['bess_soc']}
                control_signals = mpc._get_safe_fallback_control(data, initial_state)
                control_signals = [[float(control_signals[i, 0]), float(control_signals[i, 1])]
                                   for i in range(mpc.n_buildings)]
                observation, reward, done, info = env.step(control_signals)
                mpc.log_metrics(control_signals, reward)
                current_time += config['simulation_step']
                step_count += 1
            except:
                print("Fallback failed. Stopping simulation.")
                break

    # Final results
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Total Cost: {env.episode_cost:.2f} €")
    print(f"  Total Reward: {env.episode_reward:.2f}")
    print(f"  Self Sufficiency: {env.self_sufficiency:.2f}%")
    print(f"  Self Consumption: {env.self_consumption:.2f}%")
    print(f"  Total Steps: {step_count}")

    # Final plot
    mpc.plot_training_curves()

    # Save summary
    summary = pd.DataFrame([{
        'Test number': test_number,
        'Prediction horizon': config['prediction_horizon'],
        'Total cost': env.episode_cost,
        'Self sufficiency': env.self_sufficiency,
        'Self consumption': env.self_consumption,
        'Steps': step_count
    }])

    summary_path = "data/result/summary_mpc_trials.csv"
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path, sep=',', decimal='.')
        summary = pd.concat([existing, summary], ignore_index=True)

    summary.to_csv(summary_path, sep=',', decimal='.', index=False)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
