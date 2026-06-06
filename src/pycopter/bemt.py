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
        if target_thrust_N < low.total_thrust_N:
            raise ValueError(
                "Target thrust is below the minimum collective trim bracket."
            )
        if target_thrust_N > high.total_thrust_N:
            raise ValueError(
                "Target thrust is above the maximum collective trim bracket."
            )

        collective_low = self.settings.min_collective_deg
        collective_high = self.settings.max_collective_deg
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
            if collective_high - collective_low <= self.settings.collective_tolerance_deg:
                return best
            if thrust_error > 0.0:
                collective_high = collective_mid
            else:
                collective_low = collective_mid
        return best

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

    if coaxial_spec.trim_mode == "equal_thrust":
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
        interference_power_delta_W=interference_delta_W,
        interference_loss_ratio=interference_loss_ratio,
        lower_external_velocity_mean_m_s=lower_external_mean,
        wake_radius_m=wake_radius,
        wake_velocity_m_s=wake_velocity,
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
