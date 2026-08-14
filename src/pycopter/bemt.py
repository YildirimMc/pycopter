"""Hover BEMT solver with radial blade loads and coaxial interference."""

from __future__ import annotations

from dataclasses import replace
from math import acos, atan2, cos, exp, pi, sin, sqrt

import numpy as np

from .models import (
    CoaxialHoverResult,
    CoaxialSpec,
    ElementLoad,
    ExternalVelocityProfile,
    GRAVITY_M_S2,
    HoverResult,
    HoverSolverSettings,
    OperatingPoint,
    RotorSpec,
)
from .polars import PolarProvider, XfoilPolarProvider


class HoverSolver:
    """
    Blade-element/momentum hover solver.

    The equations keep full inflow-angle trigonometry and include drag in the
    sectional thrust term, matching the BEMT structure described in NASA hover
    work on laminar rotor performance. Momentum balance is solved at each
    annulus and integrated into per-blade loads and rotor totals.
    """

    def __init__(
        self,
        polar_provider: PolarProvider | None = None,
        settings: HoverSolverSettings | None = None,
    ):
        self.polar_provider = polar_provider or XfoilPolarProvider()
        self.settings = settings or HoverSolverSettings()

    def solve(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        external_axial_velocity: ExternalVelocityProfile | None = None,
    ) -> HoverResult:
        """Solve hover for either fixed collective or target thrust trim."""
        if operating_point.trim_mode == "fixed_collective":
            return self.solve_fixed_collective(
                rotor,
                operating_point,
                operating_point.collective_pitch_deg,
                external_axial_velocity,
            )

        target_thrust_N = operating_point.required_thrust_N
        low = self.solve_fixed_collective(
            rotor,
            operating_point,
            self.settings.min_collective_deg,
            external_axial_velocity,
        )
        high = self.solve_fixed_collective(
            rotor,
            operating_point,
            self.settings.max_collective_deg,
            external_axial_velocity,
        )
        collective_low = self.settings.min_collective_deg
        collective_high = self.settings.max_collective_deg

        if target_thrust_N < low.total_thrust_N:
            collective_low, low, collective_high, high = self._expand_low_collective_bracket(
                rotor,
                operating_point,
                external_axial_velocity,
                target_thrust_N,
                collective_low,
                low,
                collective_high,
                high,
            )
        if target_thrust_N > high.total_thrust_N:
            collective_low, low, collective_high, high = self._expand_high_collective_bracket(
                rotor,
                operating_point,
                external_axial_velocity,
                target_thrust_N,
                collective_low,
                low,
                collective_high,
                high,
            )

        best = high
        for _ in range(self.settings.max_trim_iterations):
            collective_mid = 0.5 * (collective_low + collective_high)
            best = self.solve_fixed_collective(
                rotor,
                operating_point,
                collective_mid,
                external_axial_velocity,
            )
            thrust_error = best.total_thrust_N - target_thrust_N
            if abs(thrust_error) <= self.settings.thrust_tolerance * target_thrust_N:
                return best
            if thrust_error > 0.0:
                collective_high = collective_mid
                high = best
            else:
                collective_low = collective_mid
                low = best
        return min(
            (low, high, best),
            key=lambda result: abs(result.total_thrust_N - target_thrust_N),
        )

    def _expand_low_collective_bracket(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        external_axial_velocity: ExternalVelocityProfile | None,
        target_thrust_N: float,
        collective_low: float,
        low: HoverResult,
        collective_high: float,
        high: HoverResult,
    ) -> tuple[float, HoverResult, float, HoverResult]:
        floor_deg = -45.0
        step_deg = max(5.0, collective_high - collective_low)
        while target_thrust_N < low.total_thrust_N and collective_low > floor_deg:
            collective_high = collective_low
            high = low
            collective_low = max(floor_deg, collective_low - step_deg)
            low = self.solve_fixed_collective(
                rotor,
                operating_point,
                collective_low,
                external_axial_velocity,
            )
            step_deg *= 1.5

        if target_thrust_N < low.total_thrust_N:
            raise ValueError(
                "Target thrust is below the achievable collective trim range."
            )
        return collective_low, low, collective_high, high

    def _expand_high_collective_bracket(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        external_axial_velocity: ExternalVelocityProfile | None,
        target_thrust_N: float,
        collective_low: float,
        low: HoverResult,
        collective_high: float,
        high: HoverResult,
    ) -> tuple[float, HoverResult, float, HoverResult]:
        ceiling_deg = 45.0
        step_deg = max(5.0, collective_high - collective_low)
        while target_thrust_N > high.total_thrust_N and collective_high < ceiling_deg:
            collective_low = collective_high
            low = high
            collective_high = min(ceiling_deg, collective_high + step_deg)
            high = self.solve_fixed_collective(
                rotor,
                operating_point,
                collective_high,
                external_axial_velocity,
            )
            step_deg *= 1.5

        if target_thrust_N > high.total_thrust_N:
            raise ValueError(
                "Target thrust is above the achievable collective trim range."
            )
        return collective_low, low, collective_high, high

    def solve_fixed_collective(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        collective_pitch_deg: float,
        external_axial_velocity: ExternalVelocityProfile | None = None,
    ) -> HoverResult:
        """Solve all radial elements at a prescribed collective pitch."""
        r_start = max(rotor.root_radius_m, rotor.stations[0].r_over_R * rotor.radius_m)
        edges = np.linspace(r_start, rotor.radius_m, self.settings.blade_element_count + 1)
        self._prepare_polar_cache(rotor, operating_point, edges, external_axial_velocity)
        element_loads = []

        for left, right in zip(edges[:-1], edges[1:]):
            r_m = 0.5 * (left + right)
            dr_m = right - left
            external_velocity = (
                float(external_axial_velocity(r_m)) if external_axial_velocity else 0.0
            )
            element_loads.append(
                self._solve_element(
                    rotor,
                    operating_point,
                    collective_pitch_deg,
                    r_m,
                    dr_m,
                    external_velocity,
                )
            )

        return self._integrate_result(rotor, operating_point, collective_pitch_deg, element_loads)

    def _prepare_polar_cache(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        edges: np.ndarray,
        external_axial_velocity: ExternalVelocityProfile | None,
    ) -> None:
        prepare_conditions = getattr(self.polar_provider, "prepare_conditions", None)
        if prepare_conditions is None:
            return

        induced_estimates = self._prefetch_induced_velocity_estimates(
            rotor,
            operating_point,
        )
        conditions = []
        for left, right in zip(edges[:-1], edges[1:]):
            r_m = 0.5 * (left + right)
            r_over_R = r_m / rotor.radius_m
            chord_m = rotor.chord_at(r_over_R)
            tangential_velocity = rotor.omega_rad_s * r_m
            external_velocity = (
                float(external_axial_velocity(r_m)) if external_axial_velocity else 0.0
            )
            for induced_velocity in induced_estimates:
                axial_velocity = external_velocity + induced_velocity
                v_rel = sqrt(tangential_velocity**2 + axial_velocity**2)
                reynolds = v_rel * chord_m / operating_point.kinematic_viscosity_m2_s
                mach = v_rel / operating_point.speed_of_sound_m_s
                conditions.append((rotor.airfoil_at(r_over_R), reynolds, mach))

        prepare_conditions(conditions)

    def _prefetch_induced_velocity_estimates(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
    ) -> tuple[float, ...]:
        target_thrust_N = None
        if operating_point.target_thrust_N is not None:
            target_thrust_N = operating_point.target_thrust_N
        elif operating_point.gross_mass_kg is not None:
            target_thrust_N = operating_point.gross_mass_kg * GRAVITY_M_S2

        if target_thrust_N is None:
            return (0.0,)

        ideal_induced = sqrt(
            target_thrust_N
            / (2.0 * operating_point.density_kg_m3 * rotor.disk_area_m2)
        )
        return (0.0, ideal_induced, 2.0 * ideal_induced)

    def _solve_element(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        collective_pitch_deg: float,
        r_m: float,
        dr_m: float,
        external_velocity_m_s: float,
    ) -> ElementLoad:
        residual_zero, load_zero = self._element_residual_and_load(
            rotor,
            operating_point,
            collective_pitch_deg,
            r_m,
            dr_m,
            external_velocity_m_s,
            induced_velocity_m_s=0.0,
        )
        if residual_zero <= 0.0:
            return load_zero

        high_velocity = max(0.1, 0.2 * rotor.tip_speed_m_s)
        max_high_velocity = max(high_velocity, rotor.tip_speed_m_s)
        high_residual = residual_zero
        high_load = load_zero
        for _ in range(40):
            high_residual, high_load = self._element_residual_and_load(
                rotor,
                operating_point,
                collective_pitch_deg,
                r_m,
                dr_m,
                external_velocity_m_s,
                induced_velocity_m_s=high_velocity,
            )
            if high_residual <= 0.0:
                break
            if high_velocity >= max_high_velocity:
                return high_load
            high_velocity = min(high_velocity * 2.0, max_high_velocity)
        else:
            return high_load

        low_velocity = 0.0
        low_residual = residual_zero
        best_load = high_load
        for _ in range(60):
            mid_velocity = 0.5 * (low_velocity + high_velocity)
            mid_residual, best_load = self._element_residual_and_load(
                rotor,
                operating_point,
                collective_pitch_deg,
                r_m,
                dr_m,
                external_velocity_m_s,
                induced_velocity_m_s=mid_velocity,
            )
            if abs(mid_residual) <= 1e-8:
                return best_load
            if mid_residual > 0.0:
                low_velocity = mid_velocity
                low_residual = mid_residual
            else:
                high_velocity = mid_velocity
            if high_velocity - low_velocity <= max(1e-6, 1e-6 * high_velocity):
                return best_load
        return best_load

    def _element_residual_and_load(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        collective_pitch_deg: float,
        r_m: float,
        dr_m: float,
        external_velocity_m_s: float,
        induced_velocity_m_s: float,
    ) -> tuple[float, ElementLoad]:
        load = self._build_element_load(
            rotor,
            operating_point,
            collective_pitch_deg,
            r_m,
            dr_m,
            external_velocity_m_s,
            induced_velocity_m_s,
        )
        momentum_thrust = (
            4.0
            * pi
            * operating_point.density_kg_m3
            * load.loss_factor
            * r_m
            * induced_velocity_m_s
            * (external_velocity_m_s + induced_velocity_m_s)
            * dr_m
        )
        blade_element_thrust = rotor.num_blades * load.dT_N
        return blade_element_thrust - momentum_thrust, load

    def _build_element_load(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        collective_pitch_deg: float,
        r_m: float,
        dr_m: float,
        external_velocity_m_s: float,
        induced_velocity_m_s: float,
    ) -> ElementLoad:
        r_over_R = r_m / rotor.radius_m
        chord_m = rotor.chord_at(r_over_R)
        twist_deg = rotor.twist_at(r_over_R)
        pitch_axis_frac = rotor.pitch_axis_at(r_over_R)
        tangential_velocity = rotor.omega_rad_s * r_m
        axial_velocity = external_velocity_m_s + induced_velocity_m_s
        phi_rad = atan2(axial_velocity, tangential_velocity)
        phi_deg = float(np.rad2deg(phi_rad))
        alpha_deg = collective_pitch_deg + twist_deg - phi_deg
        v_rel = sqrt(tangential_velocity**2 + axial_velocity**2)
        reynolds = v_rel * chord_m / operating_point.kinematic_viscosity_m2_s
        mach = v_rel / operating_point.speed_of_sound_m_s
        coeffs = self.polar_provider.get_coefficients(
            rotor.airfoil_at(r_over_R),
            alpha_deg,
            reynolds,
            mach,
        )
        loss_factor = self._loss_factor(rotor, r_m, phi_rad)
        dynamic_pressure = 0.5 * operating_point.density_kg_m3 * v_rel**2
        lift_per_m = dynamic_pressure * chord_m * coeffs.cl
        drag_per_m = dynamic_pressure * chord_m * coeffs.cd
        normal_force_per_m = lift_per_m * cos(phi_rad) - drag_per_m * sin(phi_rad)
        tangential_force_per_m = lift_per_m * sin(phi_rad) + drag_per_m * cos(phi_rad)

        dL_N = lift_per_m * dr_m
        dD_N = drag_per_m * dr_m
        dT_N = normal_force_per_m * dr_m
        dQ_Nm = r_m * tangential_force_per_m * dr_m
        dP_W = rotor.omega_rad_s * dQ_Nm
        pitch_moment_Nm = (
            dynamic_pressure * chord_m**2 * coeffs.cm * dr_m
            + dT_N * chord_m * (0.25 - pitch_axis_frac)
        )

        return ElementLoad(
            r_m=r_m,
            r_over_R=r_over_R,
            dr_m=dr_m,
            chord_m=chord_m,
            twist_deg=twist_deg,
            collective_deg=collective_pitch_deg,
            phi_deg=phi_deg,
            alpha_deg=alpha_deg,
            reynolds=reynolds,
            mach=mach,
            cl=coeffs.cl,
            cd=coeffs.cd,
            cm=coeffs.cm,
            alpha_clamped=coeffs.alpha_clamped,
            loss_factor=loss_factor,
            induced_velocity_m_s=induced_velocity_m_s,
            external_axial_velocity_m_s=external_velocity_m_s,
            dL_N=dL_N,
            dD_N=dD_N,
            dT_N=dT_N,
            dQ_Nm=dQ_Nm,
            dP_W=dP_W,
            normal_force_N_per_m=normal_force_per_m,
            tangential_force_N_per_m=tangential_force_per_m,
            pitch_moment_Nm=pitch_moment_Nm,
        )

    def _loss_factor(self, rotor: RotorSpec, r_m: float, phi_rad: float) -> float:
        if self.settings.tip_loss_model == "none" and self.settings.root_loss_model == "none":
            return 1.0

        sin_phi = abs(sin(phi_rad))
        if sin_phi < 1e-6:
            return 1.0

        factors = []
        if self.settings.tip_loss_model == "prandtl":
            f_tip = rotor.num_blades * (rotor.radius_m - r_m) / (2.0 * r_m * sin_phi)
            factors.append(self._prandtl_factor(f_tip))
        if self.settings.root_loss_model == "prandtl":
            root_radius = max(rotor.root_radius_m, rotor.stations[0].r_over_R * rotor.radius_m)
            f_root = rotor.num_blades * max(r_m - root_radius, 0.0) / (2.0 * r_m * sin_phi)
            factors.append(self._prandtl_factor(f_root))

        if not factors:
            return 1.0
        return max(self.settings.min_loss_factor, float(np.prod(factors)))

    def _prandtl_factor(self, f_value: float) -> float:
        if f_value <= 0.0:
            return self.settings.min_loss_factor
        exponent = exp(-min(f_value, 700.0))
        return float((2.0 / pi) * acos(exponent))

    def _integrate_result(
        self,
        rotor: RotorSpec,
        operating_point: OperatingPoint,
        collective_pitch_deg: float,
        element_loads: list[ElementLoad],
    ) -> HoverResult:
        per_blade_thrust_N = sum(load.dT_N for load in element_loads)
        total_thrust_N = rotor.num_blades * per_blade_thrust_N

        induced_power_per_blade = sum(
            rotor.omega_rad_s
            * load.r_m
            * (load.dL_N * sin(np.deg2rad(load.phi_deg)))
            for load in element_loads
        )
        profile_power_per_blade = sum(
            rotor.omega_rad_s
            * load.r_m
            * (load.dD_N * cos(np.deg2rad(load.phi_deg)))
            for load in element_loads
        )
        induced_power_W = (
            self.settings.induced_power_factor
            * rotor.num_blades
            * induced_power_per_blade
        )
        profile_power_W = rotor.num_blades * profile_power_per_blade
        power_W = induced_power_W + profile_power_W
        total_torque_Nm = power_W / rotor.omega_rad_s if rotor.omega_rad_s else 0.0
        per_blade_torque_Nm = total_torque_Nm / rotor.num_blades
        aircraft_yaw_torque_Nm = -rotor.rotation_direction * total_torque_Nm

        positive_thrust = max(total_thrust_N, 0.0)
        ideal_power_W = (
            positive_thrust**1.5
            / sqrt(2.0 * operating_point.density_kg_m3 * rotor.disk_area_m2)
            if positive_thrust > 0.0
            else 0.0
        )
        figure_of_merit = ideal_power_W / power_W if power_W > 0.0 else 0.0
        denominator = operating_point.density_kg_m3 * pi * rotor.radius_m**4 * rotor.omega_rad_s**2
        ct = total_thrust_N / denominator if denominator else 0.0
        cp_denominator = operating_point.density_kg_m3 * pi * rotor.radius_m**5 * rotor.omega_rad_s**3
        cp = power_W / cp_denominator if cp_denominator else 0.0

        thrust_weights = np.array([max(load.dT_N, 0.0) for load in element_loads])
        if thrust_weights.sum() > 0.0:
            mean_induced = float(
                np.average(
                    [load.induced_velocity_m_s for load in element_loads],
                    weights=thrust_weights,
                )
            )
            mean_loss = float(
                np.average([load.loss_factor for load in element_loads], weights=thrust_weights)
            )
        else:
            mean_induced = float(np.mean([load.induced_velocity_m_s for load in element_loads]))
            mean_loss = float(np.mean([load.loss_factor for load in element_loads]))

        root_flap = sum(load.dT_N * load.r_m for load in element_loads)
        root_lag = sum(load.dQ_Nm for load in element_loads)
        pitch_moment = sum(load.pitch_moment_Nm for load in element_loads)

        return HoverResult(
            collective_pitch_deg=collective_pitch_deg,
            total_thrust_N=total_thrust_N,
            per_blade_thrust_N=per_blade_thrust_N,
            total_torque_Nm=total_torque_Nm,
            per_blade_torque_Nm=per_blade_torque_Nm,
            aircraft_yaw_torque_Nm=aircraft_yaw_torque_Nm,
            power_W=power_W,
            induced_power_W=induced_power_W,
            profile_power_W=profile_power_W,
            ideal_power_W=ideal_power_W,
            figure_of_merit=figure_of_merit,
            mean_induced_velocity_m_s=mean_induced,
            ct=ct,
            cp=cp,
            solidity=rotor.solidity,
            mean_loss_factor=mean_loss,
            root_flap_bending_moment_Nm_per_blade=root_flap,
            root_lag_moment_Nm_per_blade=root_lag,
            aerodynamic_pitching_moment_Nm_per_blade=pitch_moment,
            element_loads=element_loads,
        )


def solve_coaxial_hover(
    coaxial_spec: CoaxialSpec,
    operating_point: OperatingPoint,
    polar_provider: PolarProvider | None = None,
    settings: HoverSolverSettings | None = None,
) -> CoaxialHoverResult:
    """Solve a first-order coaxial hover case with upper-wake lower-rotor inflow."""
    solver = HoverSolver(polar_provider, settings)
    lower_rotor = coaxial_spec.resolved_lower_rotor

    if coaxial_spec.trim_mode == "torque_balance":
        (
            upper,
            lower,
            isolated_upper,
            isolated_lower,
            wake_radius,
            wake_velocity,
        ) = _solve_torque_balanced_coaxial_hover(
            solver,
            coaxial_spec,
            operating_point,
            lower_rotor,
        )
    elif coaxial_spec.trim_mode == "equal_thrust":
        target_each = operating_point.required_thrust_N / 2.0
        rotor_point = replace(operating_point, target_thrust_N=target_each, gross_mass_kg=None)
        upper = solver.solve(coaxial_spec.upper_rotor, rotor_point)
        isolated_upper = upper
        isolated_lower = solver.solve(lower_rotor, rotor_point)
        wake_radius, wake_velocity = _upper_wake_at_lower_rotor(
            coaxial_spec.upper_rotor,
            upper,
            coaxial_spec.spacing_ratio,
        )

        def lower_external_velocity(r_m: float) -> float:
            return wake_velocity if r_m <= wake_radius else 0.0

        lower = solver.solve(lower_rotor, rotor_point, lower_external_velocity)
    else:
        upper_point = replace(operating_point, trim_mode="fixed_collective")
        upper = solver.solve(coaxial_spec.upper_rotor, upper_point)
        wake_radius, wake_velocity = _upper_wake_at_lower_rotor(
            coaxial_spec.upper_rotor,
            upper,
            coaxial_spec.spacing_ratio,
        )

        def lower_external_velocity(r_m: float) -> float:
            return wake_velocity if r_m <= wake_radius else 0.0

        lower_point = replace(
            operating_point,
            trim_mode="fixed_collective",
            collective_pitch_deg=(
                operating_point.collective_pitch_deg
                + coaxial_spec.lower_collective_offset_deg
            ),
        )
        lower = solver.solve(lower_rotor, lower_point, lower_external_velocity)
        isolated_upper = solver.solve(coaxial_spec.upper_rotor, upper_point)
        isolated_lower = solver.solve(lower_rotor, lower_point)

    total_power_W = upper.power_W + lower.power_W
    net_aircraft_yaw_torque_Nm = (
        upper.aircraft_yaw_torque_Nm + lower.aircraft_yaw_torque_Nm
    )
    isolated_power_W = isolated_upper.power_W + isolated_lower.power_W
    interference_delta_W = total_power_W - isolated_power_W
    interference_loss_ratio = (
        interference_delta_W / isolated_power_W if isolated_power_W > 0.0 else 0.0
    )
    lower_external_mean = (
        float(np.mean([load.external_axial_velocity_m_s for load in lower.element_loads]))
        if lower.element_loads
        else 0.0
    )

    return CoaxialHoverResult(
        upper=upper,
        lower=lower,
        isolated_upper=isolated_upper,
        isolated_lower=isolated_lower,
        total_thrust_N=upper.total_thrust_N + lower.total_thrust_N,
        total_power_W=total_power_W,
        net_aircraft_yaw_torque_Nm=net_aircraft_yaw_torque_Nm,
        interference_power_delta_W=interference_delta_W,
        interference_loss_ratio=interference_loss_ratio,
        lower_external_velocity_mean_m_s=lower_external_mean,
        wake_radius_m=wake_radius,
        wake_velocity_m_s=wake_velocity,
    )


def _solve_torque_balanced_coaxial_hover(
    solver: HoverSolver,
    coaxial_spec: CoaxialSpec,
    operating_point: OperatingPoint,
    lower_rotor: RotorSpec,
) -> tuple[HoverResult, HoverResult, HoverResult, HoverResult, float, float]:
    upper_guess, lower_guess = _equal_thrust_collective_guess(
        solver,
        coaxial_spec,
        operating_point,
        lower_rotor,
    )
    upper, lower, wake_radius, wake_velocity = _solve_collective_pair_for_trim(
        solver,
        coaxial_spec,
        operating_point,
        lower_rotor,
        upper_guess.collective_pitch_deg,
        lower_guess.collective_pitch_deg,
        apply_upper_wake=True,
    )
    isolated_upper, isolated_lower, _, _ = _solve_collective_pair_for_trim(
        solver,
        coaxial_spec,
        operating_point,
        lower_rotor,
        upper_guess.collective_pitch_deg,
        upper_guess.collective_pitch_deg,
        apply_upper_wake=False,
    )
    return upper, lower, isolated_upper, isolated_lower, wake_radius, wake_velocity


def _equal_thrust_collective_guess(
    solver: HoverSolver,
    coaxial_spec: CoaxialSpec,
    operating_point: OperatingPoint,
    lower_rotor: RotorSpec,
) -> tuple[HoverResult, HoverResult]:
    target_each = operating_point.required_thrust_N / 2.0
    rotor_point = replace(operating_point, target_thrust_N=target_each, gross_mass_kg=None)
    upper = solver.solve(coaxial_spec.upper_rotor, rotor_point)
    wake_radius, wake_velocity = _upper_wake_at_lower_rotor(
        coaxial_spec.upper_rotor,
        upper,
        coaxial_spec.spacing_ratio,
    )

    def lower_external_velocity(r_m: float) -> float:
        return wake_velocity if r_m <= wake_radius else 0.0

    lower = solver.solve(lower_rotor, rotor_point, lower_external_velocity)
    return upper, lower


def _solve_collective_pair_for_trim(
    solver: HoverSolver,
    coaxial_spec: CoaxialSpec,
    operating_point: OperatingPoint,
    lower_rotor: RotorSpec,
    upper_collective_deg: float,
    lower_collective_deg: float,
    *,
    apply_upper_wake: bool,
) -> tuple[HoverResult, HoverResult, float, float]:
    target_thrust_N = operating_point.required_thrust_N
    lower_bound = -45.0
    upper_bound = 45.0
    step_deg = 0.25
    best = _evaluate_collective_pair(
        solver,
        coaxial_spec,
        operating_point,
        lower_rotor,
        upper_collective_deg,
        lower_collective_deg,
        apply_upper_wake=apply_upper_wake,
    )
    best_norm = _coaxial_trim_norm(best, target_thrust_N)

    for _ in range(min(solver.settings.max_trim_iterations, 30)):
        thrust_error, yaw_error = _coaxial_trim_errors(best, target_thrust_N)
        torque_scale = max(
            abs(best[0].total_torque_Nm) + abs(best[1].total_torque_Nm),
            1.0,
        )
        if (
            abs(thrust_error) <= solver.settings.thrust_tolerance * target_thrust_N
            and abs(yaw_error) <= max(1e-4, 1e-3 * torque_scale)
        ):
            return best

        upper_step = _evaluate_collective_pair(
            solver,
            coaxial_spec,
            operating_point,
            lower_rotor,
            min(upper_bound, upper_collective_deg + step_deg),
            lower_collective_deg,
            apply_upper_wake=apply_upper_wake,
        )
        lower_step = _evaluate_collective_pair(
            solver,
            coaxial_spec,
            operating_point,
            lower_rotor,
            upper_collective_deg,
            min(upper_bound, lower_collective_deg + step_deg),
            apply_upper_wake=apply_upper_wake,
        )
        upper_errors = _coaxial_trim_errors(upper_step, target_thrust_N)
        lower_errors = _coaxial_trim_errors(lower_step, target_thrust_N)

        a = (upper_errors[0] - thrust_error) / step_deg
        b = (lower_errors[0] - thrust_error) / step_deg
        c = (upper_errors[1] - yaw_error) / step_deg
        d = (lower_errors[1] - yaw_error) / step_deg
        determinant = a * d - b * c
        if abs(determinant) < 1e-9:
            break

        delta_upper = (-thrust_error * d + b * yaw_error) / determinant
        delta_lower = (c * thrust_error - a * yaw_error) / determinant
        max_step = 4.0
        largest = max(abs(delta_upper), abs(delta_lower), 1.0)
        if largest > max_step:
            scale = max_step / largest
            delta_upper *= scale
            delta_lower *= scale

        accepted = None
        accepted_norm = None
        for damping in (1.0, 0.5, 0.25, 0.1):
            candidate_upper = min(
                upper_bound,
                max(lower_bound, upper_collective_deg + damping * delta_upper),
            )
            candidate_lower = min(
                upper_bound,
                max(lower_bound, lower_collective_deg + damping * delta_lower),
            )
            candidate = _evaluate_collective_pair(
                solver,
                coaxial_spec,
                operating_point,
                lower_rotor,
                candidate_upper,
                candidate_lower,
                apply_upper_wake=apply_upper_wake,
            )
            candidate_norm = _coaxial_trim_norm(candidate, target_thrust_N)
            if candidate_norm < best_norm or accepted is None:
                accepted = candidate
                accepted_norm = candidate_norm
            if candidate_norm < best_norm:
                break

        if accepted is None or accepted_norm is None or accepted_norm >= best_norm:
            break
        best = accepted
        best_norm = accepted_norm
        upper_collective_deg = best[0].collective_pitch_deg
        lower_collective_deg = best[1].collective_pitch_deg

    thrust_error, yaw_error = _coaxial_trim_errors(best, target_thrust_N)
    torque_scale = max(abs(best[0].total_torque_Nm) + abs(best[1].total_torque_Nm), 1.0)
    if (
        abs(thrust_error) > 5.0 * solver.settings.thrust_tolerance * target_thrust_N
        or abs(yaw_error) > max(1e-3, 5e-3 * torque_scale)
    ):
        raise ValueError("Torque-balanced coaxial trim did not converge.")
    return best


def _evaluate_collective_pair(
    solver: HoverSolver,
    coaxial_spec: CoaxialSpec,
    operating_point: OperatingPoint,
    lower_rotor: RotorSpec,
    upper_collective_deg: float,
    lower_collective_deg: float,
    *,
    apply_upper_wake: bool,
) -> tuple[HoverResult, HoverResult, float, float]:
    upper = solver.solve_fixed_collective(
        coaxial_spec.upper_rotor,
        _fixed_collective_point(operating_point, upper_collective_deg),
        upper_collective_deg,
    )
    if apply_upper_wake:
        wake_radius, wake_velocity = _upper_wake_at_lower_rotor(
            coaxial_spec.upper_rotor,
            upper,
            coaxial_spec.spacing_ratio,
        )
    else:
        wake_radius, wake_velocity = coaxial_spec.upper_rotor.radius_m, 0.0

    def lower_external_velocity(r_m: float) -> float:
        return wake_velocity if apply_upper_wake and r_m <= wake_radius else 0.0

    lower = solver.solve_fixed_collective(
        lower_rotor,
        _fixed_collective_point(operating_point, lower_collective_deg),
        lower_collective_deg,
        lower_external_velocity,
    )
    return upper, lower, wake_radius, wake_velocity


def _coaxial_trim_errors(
    pair: tuple[HoverResult, HoverResult, float, float],
    target_thrust_N: float,
) -> tuple[float, float]:
    upper, lower, _, _ = pair
    return (
        upper.total_thrust_N + lower.total_thrust_N - target_thrust_N,
        upper.aircraft_yaw_torque_Nm + lower.aircraft_yaw_torque_Nm,
    )


def _coaxial_trim_norm(
    pair: tuple[HoverResult, HoverResult, float, float],
    target_thrust_N: float,
) -> float:
    thrust_error, yaw_error = _coaxial_trim_errors(pair, target_thrust_N)
    upper, lower, _, _ = pair
    torque_scale = max(abs(upper.total_torque_Nm) + abs(lower.total_torque_Nm), 1.0)
    return abs(thrust_error) / max(target_thrust_N, 1e-9) + abs(yaw_error) / torque_scale


def _fixed_collective_point(
    operating_point: OperatingPoint,
    collective_pitch_deg: float,
) -> OperatingPoint:
    return replace(
        operating_point,
        trim_mode="fixed_collective",
        collective_pitch_deg=collective_pitch_deg,
    )


def _upper_wake_at_lower_rotor(
    upper_rotor: RotorSpec,
    upper_result: HoverResult,
    spacing_ratio: float,
) -> tuple[float, float]:
    """
    Approximate the upper wake at the lower rotor using actuator-disk continuity.

    The velocity transitions from disk induced velocity toward the far-wake
    value 2*vi. The contracted wake radius follows A*wake_velocity = A0*vi.
    """
    induced = max(upper_result.mean_induced_velocity_m_s, 0.0)
    if induced <= 0.0:
        return upper_rotor.radius_m, 0.0

    spacing_m = spacing_ratio * upper_rotor.radius_m
    wake_velocity = induced * (1.0 + spacing_m / (spacing_m + upper_rotor.radius_m))
    wake_radius = upper_rotor.radius_m * sqrt(induced / wake_velocity)
    return wake_radius, wake_velocity
