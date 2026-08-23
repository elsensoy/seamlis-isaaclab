import casadi as ca

from safe_control.position_control.mpc_cbf import MPCCBF as _BaseMPCCBF


class SeamlisMPCCBF(_BaseMPCCBF):
    """Local MPC-CBF extension with dynamic-obstacle CBF for inter-agent safety."""

    # Static obstacles use 0 (circle) / 1 (superellipse). We reserve >1.5 for dynamic circles.
    _DYNAMIC_OBS_FLAG_THRESHOLD = 1.5

    def agent_barrier_dyn_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        """Discrete-time CBF terms for moving circular obstacles with constant obstacle velocity."""
        model = self.robot_spec.get("model", "")
        dt = float(self.dt)

        # Obstacle state: [ox, oy, r, vx, vy, _, dynamic_flag]
        ox = obs[0]
        oy = obs[1]
        r_obs = obs[2]
        vxo = obs[3]
        vyo = obs[4]
        d_min = robot_radius + r_obs

        if model == "SingleIntegrator2D":
            # x_{k+1} = x_k + dt * u_k
            rx = x_k[0, 0] - ox
            ry = x_k[1, 0] - oy
            rx1 = (x_k[0, 0] + dt * u_k[0, 0]) - (ox + dt * vxo)
            ry1 = (x_k[1, 0] + dt * u_k[1, 0]) - (oy + dt * vyo)

            h_k = rx * rx + ry * ry - beta * d_min * d_min
            h_k1 = rx1 * rx1 + ry1 * ry1 - beta * d_min * d_min
            d_h = h_k1 - h_k
            return h_k, d_h

        if model == "DoubleIntegrator2D":
            # x_{k+1} = x_k + dt * v_k, v_{k+1} = v_k + dt * a_k
            rx = x_k[0, 0] - ox
            ry = x_k[1, 0] - oy
            rvx = x_k[2, 0] - vxo
            rvy = x_k[3, 0] - vyo

            rx1 = rx + dt * rvx
            ry1 = ry + dt * rvy
            rx2 = rx + 2.0 * dt * rvx + (dt * dt) * u_k[0, 0]
            ry2 = ry + 2.0 * dt * rvy + (dt * dt) * u_k[1, 0]

            h_k = rx * rx + ry * ry - beta * d_min * d_min
            h_k1 = rx1 * rx1 + ry1 * ry1 - beta * d_min * d_min
            h_k2 = rx2 * rx2 + ry2 * ry2 - beta * d_min * d_min

            d_h = h_k1 - h_k
            dd_h = h_k2 - 2.0 * h_k1 + h_k
            return h_k, d_h, dd_h

        # Fallback for other models (use static implementation).
        return self.robot.agent_barrier_dt(x_k, u_k, obs, robot_radius)

    def compute_cbf_constraint(self, _x, _u, _obs):
        static_cbf = super().compute_cbf_constraint(_x, _u, _obs)

        model = self.robot_spec.get("model", "")
        if model not in ["SingleIntegrator2D", "DoubleIntegrator2D"]:
            return static_cbf

        is_dynamic_obs = _obs[6] > self._DYNAMIC_OBS_FLAG_THRESHOLD

        if model == "SingleIntegrator2D":
            _alpha = self.model.tvp["alpha"]
            h_k_dyn, d_h_dyn = self.agent_barrier_dyn_dt(
                _x, _u, _obs, self.robot.robot_radius
            )
            dyn_cbf = d_h_dyn + _alpha * h_k_dyn
            return ca.if_else(is_dynamic_obs, dyn_cbf, static_cbf)

        _alpha1 = self.model.tvp["alpha1"]
        _alpha2 = self.model.tvp["alpha2"]
        h_k_dyn, d_h_dyn, dd_h_dyn = self.agent_barrier_dyn_dt(
            _x, _u, _obs, self.robot.robot_radius
        )
        dyn_cbf = dd_h_dyn + (_alpha1 + _alpha2) * d_h_dyn + _alpha1 * _alpha2 * h_k_dyn
        return ca.if_else(is_dynamic_obs, dyn_cbf, static_cbf)
