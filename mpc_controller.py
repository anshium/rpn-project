import numpy as np
import cvxopt

class MPCController:
    def __init__(self, robot_radius, vxmax, vymax,
                 planning_horizon=20, control_horizon=5, del_t=0.1,
                 initial_vx_guess=None, initial_vy_guess=None):
        self.robot_radius = robot_radius 
        self.vxmax = vxmax
        self.vymax = vymax

        self.planning_horizon = planning_horizon
        self.control_horizon = control_horizon
        self.del_t = del_t

        if initial_vx_guess is None:
            self.vx_guess = np.zeros(planning_horizon)
        else:
            self.vx_guess = np.array(initial_vx_guess, dtype=float) # Ensure float
            if np.any(np.isnan(self.vx_guess)) or np.any(np.isinf(self.vx_guess)):
                print("Warning: initial_vx_guess contains NaN/Inf. Resetting to zeros.")
                self.vx_guess = np.zeros(planning_horizon)
            if len(self.vx_guess) != planning_horizon:
                print(f"Warning: initial_vx_guess length {len(self.vx_guess)} != planning_horizon {planning_horizon}. Adjusting.")
                self.vx_guess = np.resize(self.vx_guess, planning_horizon)


        if initial_vy_guess is None:
            self.vy_guess = np.zeros(planning_horizon)
        else:
            self.vy_guess = np.array(initial_vy_guess, dtype=float) # Ensure float
            if np.any(np.isnan(self.vy_guess)) or np.any(np.isinf(self.vy_guess)):
                print("Warning: initial_vy_guess contains NaN/Inf. Resetting to zeros.")
                self.vy_guess = np.zeros(planning_horizon)
            if len(self.vy_guess) != planning_horizon:
                print(f"Warning: initial_vy_guess length {len(self.vy_guess)} != planning_horizon {planning_horizon}. Adjusting.")
                self.vy_guess = np.resize(self.vy_guess, planning_horizon)
        
        self.current_obstacles_pos = [] 
        self.current_obstacles_rad = [] 

        self.optimized_vx = np.copy(self.vx_guess)
        self.optimized_vy = np.copy(self.vy_guess)

    def _p_constructor(self):
        A = np.ones((self.planning_horizon, 1))
        P_block = 2 * self.del_t**2 * np.matmul(A, A.T)
        P_block = (P_block + P_block.T) / 2 
        net_P = np.kron(np.eye(2), P_block)
        q_gen_col_vector = 2 * self.del_t * A 
        return net_P, q_gen_col_vector

    def _obstacle_constraint_terms(self, current_pos_x, current_pos_y, future_step_idx, obst_idx, current_vx_guess, current_vy_guess):
        x_g = current_pos_x + np.sum(current_vx_guess[:future_step_idx + 1] * self.del_t)
        y_g = current_pos_y + np.sum(current_vy_guess[:future_step_idx + 1] * self.del_t)

        obst_x = self.current_obstacles_pos[obst_idx][0]
        obst_y = self.current_obstacles_pos[obst_idx][1]
        
        f_at_uk = (x_g - obst_x)**2 + (y_g - obst_y)**2
        
        grad_fx_coeffs = np.zeros(self.planning_horizon)
        grad_fy_coeffs = np.zeros(self.planning_horizon)

        grad_fx_coeffs[:future_step_idx + 1] = 2 * (x_g - obst_x) * self.del_t
        grad_fy_coeffs[:future_step_idx + 1] = 2 * (y_g - obst_y) * self.del_t
        
        grad_f_uk = np.concatenate((grad_fx_coeffs, grad_fy_coeffs)) 
        
        grad_f_uk_dot_uk = np.dot(grad_f_uk, np.concatenate((current_vx_guess, current_vy_guess)))
        
        return grad_f_uk, f_at_uk, grad_f_uk_dot_uk

    def compute_control_actions(self, current_pos, goal_pos, obstacles_info):
        self.current_obstacles_pos = [obs['position'] for obs in obstacles_info]
        self.current_obstacles_rad = [obs['radius'] for obs in obstacles_info]

        P_cvx, q_gen_col_vector = self._p_constructor()
        P_cvx += 1e-6 * np.eye(P_cvx.shape[0]) 

        q_cvx = np.concatenate(
            ((current_pos[0] - goal_pos[0]) * q_gen_col_vector.flatten(),
             (current_pos[1] - goal_pos[1]) * q_gen_col_vector.flatten()),
            axis=0
        )
        
        P = cvxopt.matrix(P_cvx, tc='d')
        q = cvxopt.matrix(q_cvx, tc='d')

        G_list = []
        h_list = []

        id_N = np.eye(self.planning_horizon)
        zeros_N_N = np.zeros((self.planning_horizon, self.planning_horizon))
        
        G_vx_upper = np.hstack([id_N, zeros_N_N])
        G_vx_lower = np.hstack([-id_N, zeros_N_N])
        G_vy_upper = np.hstack([zeros_N_N, id_N])
        G_vy_lower = np.hstack([zeros_N_N, -id_N])

        G_list.extend([G_vx_upper, G_vx_lower, G_vy_upper, G_vy_lower])
        
        h_vx_limits = self.vxmax * np.ones(self.planning_horizon)
        h_vy_limits = self.vymax * np.ones(self.planning_horizon)
        h_list.extend([h_vx_limits, h_vx_limits, h_vy_limits, h_vy_limits])

        safety_margin = 0.1 
        for obs_idx in range(len(self.current_obstacles_pos)):
            min_dist_sq = (self.current_obstacles_rad[obs_idx] + self.robot_radius + safety_margin)**2
            for i in range(self.planning_horizon): 
                grad_f_uk, f_at_uk, grad_f_uk_dot_uk = self._obstacle_constraint_terms(
                    current_pos[0], current_pos[1], i, obs_idx, self.vx_guess, self.vy_guess
                )
                
                G_row = -grad_f_uk.reshape(1, -1)
                h_val = f_at_uk - grad_f_uk_dot_uk - min_dist_sq
                
                G_list.append(G_row)
                h_list.append(np.array([h_val]))

        if not G_list: 
            print("Warning: G_list is empty. Using previous good plan.")
            return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]

        G_final = np.vstack(G_list)
        h_final = np.concatenate(h_list)

        G = cvxopt.matrix(G_final, tc='d')
        h = cvxopt.matrix(h_final, tc='d')
        
        initial_solver_guess = cvxopt.matrix(np.concatenate((self.vx_guess, self.vy_guess)), tc='d')

        cvxopt.solvers.options['show_progress'] = False
        
        try:
            sol = cvxopt.solvers.qp(P, q, G, h, initvals=initial_solver_guess)
            
            # --- ROBUSTNESS CHECKS FOR SOLVER OUTPUT ---
            if sol['x'] is None:
                print(f"Warning: MPC QP solver status: {sol['status']}, but sol['x'] is None. Using previous good plan.")
                return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]

            # Convert to NumPy arrays for easier NaN/Inf checking
            vx_candidate = np.array(sol['x'][:self.planning_horizon]).flatten()
            vy_candidate = np.array(sol['x'][self.planning_horizon:]).flatten()

            if np.any(np.isnan(vx_candidate)) or \
               np.any(np.isinf(vx_candidate)) or \
               np.any(np.isnan(vy_candidate)) or \
               np.any(np.isinf(vy_candidate)):
                print(f"Warning: MPC QP solver status: {sol['status']}, but solution contains NaN/Inf. Using previous good plan.")
                # print(f"NaN/Inf vx_candidate: {vx_candidate}") # Optional: for deeper debugging
                # print(f"NaN/Inf vy_candidate: {vy_candidate}")
                return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]
            
            if sol['status'] != 'optimal':
                print(f"Warning: MPC QP solver status: {sol['status']}. Using previous good plan.")
                return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]
            # --- END OF ROBUSTNESS CHECKS ---

            # If all checks passed, the solution is deemed valid
            self.optimized_vx = vx_candidate
            self.optimized_vy = vy_candidate

        except ValueError as e: 
            print(f"CVXOPT ValueError (likely infeasible or numerical issue): {e}. Using previous good plan.")
            return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]
        except TypeError as e: # CVXOPT can sometimes raise TypeError on bad inputs (e.g. if G or h have NaNs)
            print(f"CVXOPT TypeError: {e}. Using previous good plan.")
            return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]
        except ArithmeticError as e: # Catch issues like division by zero if they propagate from C layer
            print(f"CVXOPT ArithmeticError: {e}. Using previous good plan.")
            return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]


        self.vx_guess[:-self.control_horizon] = self.optimized_vx[self.control_horizon:]
        self.vx_guess[-self.control_horizon:] = self.optimized_vx[-1] 

        self.vy_guess[:-self.control_horizon] = self.optimized_vy[self.control_horizon:]
        self.vy_guess[-self.control_horizon:] = self.optimized_vy[-1] 

        return self.optimized_vx[:self.control_horizon], self.optimized_vy[:self.control_horizon]