"""Two-DOF quarter car -- the bump load case for force extraction.

WHY THIS EXISTS
---------------
`suspension.forces` turns a contact-patch load into member forces, and the
steady-state cases (1.5 g cornering, 1.2 g accel, 1.8 g braking) come straight
from T1: prescribe the acceleration, read the per-corner Fz/Fy/Fx, extract.

A BUMP has no such shortcut. Nothing in T1 or T2 knows what vertical load a
kerb strike produces, because both are steady-state solvers and a bump is a
transient. You cannot read it off a g-g diagram. So this module simulates it.

>>> THE THING THAT MAKES A BUMP CASE WRONG IF YOU SKIP IT <<<

The links do NOT carry the peak contact-patch force. The free body of the
upright + wheel + hub gives

    F_links = F_tire - m_unsprung * (a_unsprung + g)

and in a sharp bump `m_u * a_u` is a large fraction of `F_tire` -- the tire
force is busy accelerating the unsprung mass, and only the remainder reaches
the links. Applying a peak Fz on its own, with `unsprung_accel` left at zero,
therefore overstates link loads, and it does so worst exactly where the case
matters most (short, sharp inputs).

That is why `bump_case` returns the tire force AND the unsprung acceleration
AT THE SAME INSTANT, and why `to_corner_loads` sets both. `forces.CornerLoads`
already applies the d'Alembert term; this module's job is to supply an honest
pair, not a scary Fz.

WHAT THIS IS NOT
----------------
It is not T3. A quarter car has no roll, no pitch, no load transfer, no tire
relaxation, and one wheel. It answers "what vertical force does this bump
produce at this corner", which is the only question the force extractor needs
of it. The 14-DOF model (docs/07) subsumes it later; this stays useful because
a structural load case wants one corner and one number, not a lap.

MODEL
-----
Standard 2-DOF: sprung corner mass on spring+damper, unsprung mass on the tire
spring, road displacement input.

    m_s * z_s'' = -F_susp
    m_u * z_u'' =  F_susp - F_tire        (all displacements positive UP)

    F_susp = k_w*(z_u - z_s) + c(v)*(z_u' - z_s')      compression positive
    F_tire = k_t*(z_u - z_road) + c_t*(z_u' - z_road')

`F_tire` here is the tire's DEFLECTION force, POSITIVE IN EXTENSION, so the
contact-patch normal load is `static_wheel_load - F_tire`. A tire cannot pull
the road up, so that force is clamped at the static load -- IN THE INTEGRATION
as well as in what gets reported, which is the whole content of the clamp in
`_derivatives`. Past the clamp the wheel is airborne and the tire does nothing
until it lands.

Rates are AT THE WHEEL. The config stores damper rates at the DAMPER, so
`from_config` converts with the motion ratio -- c_wheel = c_damper * MR^2 and
the knee velocity divides by MR. Getting that backwards is a factor of ~MR^4
on damping, which is why it happens in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ..core.units import G

__all__ = ["QuarterCar", "QuarterCarResult", "BumpResult",
           "half_sine_bump", "step_bump", "bump_case", "bump_envelope"]


# ---------------------------------------------------------------------------
# Road inputs
# ---------------------------------------------------------------------------

def half_sine_bump(height_m: float, length_m: float, speed_m_s: float
                   ) -> Callable[[float], Tuple[float, float]]:
    """A single half-sine bump traversed at constant speed.

    Returns f(t) -> (z_road, z_road_dot).

    The half sine is the standard idealisation of a kerb or a road ripple: it
    is C0 at both ends with zero slope, so it excites the suspension without
    the infinite velocity of a step. `length_m` is the bump's LONGITUDINAL
    extent, so the duration is length/speed -- speed is what makes a bump
    severe, not height alone.
    """
    if length_m <= 0 or speed_m_s <= 0:
        raise ValueError("length and speed must be positive")
    duration = length_m / speed_m_s
    w = np.pi / duration

    def road(t: float) -> Tuple[float, float]:
        if t < 0.0 or t > duration:
            return 0.0, 0.0
        return (height_m * np.sin(w * t),
                height_m * w * np.cos(w * t))
    return road


def step_bump(height_m: float, rise_time_s: float = 1e-3
              ) -> Callable[[float], Tuple[float, float]]:
    """A step of `height_m`, smoothed over `rise_time_s`.

    The worst case a quarter car can represent. A true discontinuity has
    infinite road velocity and would make the tire damper term meaningless, so
    the rise is finite and reported -- if a result depends strongly on
    `rise_time_s`, the answer is being set by the idealisation, not the car.
    """
    if rise_time_s <= 0:
        raise ValueError("rise_time_s must be positive")

    def road(t: float) -> Tuple[float, float]:
        if t <= 0.0:
            return 0.0, 0.0
        if t >= rise_time_s:
            return height_m, 0.0
        x = t / rise_time_s
        return (height_m * 0.5 * (1.0 - np.cos(np.pi * x)),
                height_m * 0.5 * np.pi / rise_time_s * np.sin(np.pi * x))
    return road


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class QuarterCar:
    """One corner. All rates AT THE WHEEL, SI, displacements positive up."""

    sprung_mass: float
    unsprung_mass: float
    wheel_rate: float
    """Spring rate at the wheel, N/m -- spring rate * MR^2, NOT the spring."""
    tire_rate: float
    damper_bump_low: float
    damper_bump_high: float
    damper_rebound_low: float
    damper_rebound_high: float
    knee_velocity: float
    """Wheel velocity at which the damper curve changes slope, m/s."""
    tire_damping: float = 500.0
    """N/(m/s). Small and rarely measured, and it earns its place twice. A tire
    with zero damping rings at its hop frequency for far longer than a real one
    -- 65% more hop-band energy in the front corner's contact load -- and,
    because it multiplies ROAD velocity, it is a large share of the leading-edge
    peak on a sharp input: 4176 N against 3484 N at zero damping, NFR27 front,
    25 mm over 300 mm at 20 m/s.

    What it does NOT do is hold the second and third force peaks down by ringing
    the tire out, which an earlier version of this note claimed. At every speed
    in the config's bump sweep the wheel LEAVES THE GROUND (27-84 ms airborne at
    the front), and since `_derivatives` clamps the tire force an airborne tire
    exerts nothing at all -- so those later peaks are RE-CONTACT IMPACTS, not
    the tail of a ring. Tire damping still moves them (second peak 1273 -> 1074
    N at 4 m/s) by setting how hard the wheel lands, which is a different claim.
    500 is a typical passenger/race figure."""

    def damper_force(self, v_compression):
        """Damper force at the wheel, N, positive resisting compression.

        Digressive bilinear: the LOW-speed slope is the steeper one (that is
        what 'digressive' means), so below the knee the damper is stiff for
        body control and above it it blows off to let the wheel move over a
        sharp input. Getting these two the wrong way round makes a bump case
        look far worse than it is.
        """
        v = np.asarray(v_compression, dtype=float)
        vk = self.knee_velocity
        lo = np.where(v >= 0.0, self.damper_bump_low, self.damper_rebound_low)
        hi = np.where(v >= 0.0, self.damper_bump_high, self.damper_rebound_high)
        mag = np.abs(v)
        force = np.where(mag <= vk, lo * mag, lo * vk + hi * (mag - vk))
        return np.sign(v) * force

    @property
    def static_wheel_load(self) -> float:
        """Vertical load at the contact patch at rest, N."""
        return (self.sprung_mass + self.unsprung_mass) * G

    @property
    def ride_frequency_hz(self) -> float:
        k = 1.0 / (1.0 / self.wheel_rate + 1.0 / self.tire_rate)
        return float(np.sqrt(k / self.sprung_mass) / (2.0 * np.pi))

    @property
    def wheel_hop_frequency_hz(self) -> float:
        return float(np.sqrt((self.wheel_rate + self.tire_rate)
                             / self.unsprung_mass) / (2.0 * np.pi))

    # ------------------------------------------------------------------
    def _derivatives(self, state, road):
        z_s, v_s, z_u, v_u = state
        z_r, v_r = road

        f_susp = (self.wheel_rate * (z_u - z_s)
                  + self.damper_force(v_u - v_s))
        f_tire = (self.tire_rate * (z_u - z_r)
                  + self.tire_damping * (v_u - v_r))

        # >>> THE INTEGRATION HAS TO KNOW THE WHEEL LEFT THE GROUND, NOT JUST
        #     THE REPORTING. <<<
        #
        # `f_tire` is the tire's EXTENSION force, so the contact normal load is
        # `static_wheel_load - f_tire` and contact is lost the moment `f_tire`
        # reaches the static load. `simulate` has always floored the load it
        # REPORTS at that threshold; without the same one-sided clamp here the
        # integrated tire went into TENSION past it and pulled the airborne
        # wheel back down toward the road.
        #
        # Measured, NFR27 front corner, 25 mm over 300 mm at 20 m/s: the wheel
        # first lifts at 10.6 ms, and at 15.0 ms the unclamped model reported
        # a_unsprung = -28.8 g with Fz_contact = 0. An airborne wheel can only
        # be pushed down by gravity plus the suspension, which at that instant
        # is -3.8 g -- 7.6x out -- so every sample after the first lift instant
        # was fiction. The peak contact load always occurs BEFORE lift, so no
        # `bump_case` number moves; both facts are pinned by
        # test_airborne_unsprung_accel_is_bounded_by_gravity_plus_suspension and
        # test_clamp_does_not_move_the_reported_bump_peaks.
        #
        # One-sided, on the TOTAL (spring + damping) force, at exactly the
        # threshold `simulate` floors at -- so the two agree by construction.
        # Scalar `min` because every caller here integrates scalar state.
        f_tire = min(f_tire, self.static_wheel_load)

        # SIGNS, derived rather than guessed -- both were inverted first time
        # and the integration diverged to 1e30 within a few ms, which is the
        # only reason it was caught. With d = z_u - z_s positive in BUMP, the
        # suspension pushes the two masses APART: sprung UP (+f_susp), unsprung
        # DOWN (-f_susp). And with f_tire = k_t*(z_u - z_r) positive when the
        # tire is EXTENDED, the tire's force on the wheel is -f_tire.
        return np.array([v_s,
                         f_susp / self.sprung_mass,
                         v_u,
                         (-f_susp - f_tire) / self.unsprung_mass])

    def simulate(self, road: Callable[[float], Tuple[float, float]],
                 duration: float, dt: float = 2.0e-5) -> "QuarterCarResult":
        """Integrate from static equilibrium. RK4, fixed step.

        `dt` default 20 us resolves the wheel-hop mode (order 15 Hz on this
        car) with ~3000 steps per cycle. Fixed-step RK4 rather than an adaptive
        solver because the road input is only piecewise smooth, and adaptive
        steppers waste most of their effort re-detecting the same corner at the
        bump's leading edge.
        """
        n = int(np.ceil(duration / dt)) + 1
        t = np.arange(n) * dt

        # Static equilibrium: both masses at rest, springs carrying their load.
        # Working in deflection-from-static coordinates keeps gravity out of
        # the integration entirely -- it cancels exactly at t=0 and would
        # otherwise have to be re-added to every force term.
        state = np.zeros(4)
        out = np.zeros((n, 4))
        f_tire = np.zeros(n)
        a_u = np.zeros(n)

        for i in range(n):
            out[i] = state
            z_r, v_r = road(t[i])
            f_tire[i] = (self.tire_rate * (state[2] - z_r)
                         + self.tire_damping * (state[3] - v_r))
            a_u[i] = self._derivatives(state, (z_r, v_r))[3]

            if i == n - 1:
                break
            k1 = self._derivatives(state, road(t[i]))
            k2 = self._derivatives(state + 0.5 * dt * k1, road(t[i] + 0.5 * dt))
            k3 = self._derivatives(state + 0.5 * dt * k2, road(t[i] + 0.5 * dt))
            k4 = self._derivatives(state + dt * k3, road(t[i] + dt))
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Contact-patch normal load = static load MINUS the deflection force,
        # floored at 0. `f_tire` is positive when the tire is EXTENDED relative
        # to static (z_u above z_road), which is the sign that SHEDS load, so
        # this floor bites at exactly the threshold `_derivatives` clamps at --
        # they are the same statement about the same instant.
        Fz = np.maximum(self.static_wheel_load - f_tire, 0.0)

        return QuarterCarResult(
            t=t, z_sprung=out[:, 0], v_sprung=out[:, 1],
            z_unsprung=out[:, 2], v_unsprung=out[:, 3],
            Fz_contact=Fz, a_unsprung=a_u, car=self)


@dataclass
class QuarterCarResult:
    t: np.ndarray
    z_sprung: np.ndarray
    v_sprung: np.ndarray
    z_unsprung: np.ndarray
    v_unsprung: np.ndarray
    Fz_contact: np.ndarray
    """Contact-patch normal load, N. Zero means the wheel has left the ground."""
    a_unsprung: np.ndarray
    """Vertical acceleration of the unsprung mass, m/s^2, positive up."""
    car: QuarterCar

    @property
    def wheel_travel(self) -> np.ndarray:
        """Suspension deflection, m. Positive = bump (wheel up relative to body)."""
        return self.z_unsprung - self.z_sprung

    @property
    def wheel_lifted(self) -> bool:
        return bool(np.any(self.Fz_contact <= 0.0))


@dataclass
class BumpResult:
    """The peak instant of a bump, packaged for force extraction."""

    Fz_peak: float
    """Peak contact-patch normal load, N."""
    a_unsprung_at_peak: float
    """Unsprung vertical acceleration at that same instant, m/s^2."""
    t_peak: float
    load_factor: float
    """Fz_peak / static load. The '3 g bump' number, but MEASURED."""
    link_borne_force: float
    """Fz_peak - m_u*(a_u + g) AT THE CONTACT-PATCH PEAK INSTANT.

    >>> A DIAGNOSTIC, NOT A SIZING NUMBER. DO NOT SELECT ON IT. <<<

    This used to say "this, not Fz_peak, is the number that sizes a pushrod",
    and `force_matrix._bump_result` duly picked its bump speed by maximising it.
    Both were wrong for the same reason: the force reaching the links does not
    peak at the instant the contact patch does, so sampling it here reads the
    trace at the wrong moment. Measured on the NFR26 front corner, 25 mm over
    300 mm, this field against the true maximum over the same trace:

        speed     this field     true max over the trace
         5 m/s       1735 N              1852 N
        11 m/s       1964 N              2309 N
        20 m/s       1838 N              2496 N
        30 m/s       1639 N              2552 N

    So it understates by 7% at 5 m/s and 56% at 30 m/s, and -- worse than the
    magnitude -- it TURNS OVER while the truth does not, which is what made a
    moderate speed look like the design condition. Selecting on it picked
    11 m/s where the linkage is actually worst at the top of the sweep.

    What sizes a part is `ForceMatrix.envelope()`, which walks every instant of
    every speed and takes the maximum per attachment point. Keep this field for
    reading a single simulation, not for choosing between them."""
    max_wheel_travel: float
    wheel_lifted: bool
    result: QuarterCarResult

    def report(self) -> str:
        return (
            f"peak contact-patch Fz {self.Fz_peak:8.1f} N  "
            f"({self.load_factor:.2f} x static) at t = {self.t_peak * 1000:.1f} ms\n"
            f"unsprung accel at peak {self.a_unsprung_at_peak / G:+7.2f} g\n"
            f"force reaching the LINKS {self.link_borne_force:8.1f} N  "
            f"({self.link_borne_force / max(self.Fz_peak, 1e-9):.0%} of the "
            f"contact-patch peak)\n"
            f"max wheel travel {self.max_wheel_travel * 1000:6.1f} mm"
            + ("   >>> WHEEL LEFT THE GROUND <<<" if self.wheel_lifted else ""))

    def to_corner_loads(self, **kw):
        """Build a `forces.CornerLoads` for this instant.

        Sets BOTH `Fz` and `unsprung_accel` -- see the module docstring. Extra
        keyword arguments pass through, so a combined case (bump while
        cornering, say) can add Fy/Fx on top.
        """
        from ..suspension.forces import CornerLoads
        kw.setdefault("Fz", self.Fz_peak)
        kw.setdefault("unsprung_mass", self.result.car.unsprung_mass)
        kw.setdefault("unsprung_accel",
                      np.array([0.0, 0.0, self.a_unsprung_at_peak]))
        return CornerLoads(**kw)


def bump_case(car: QuarterCar,
              road: Callable[[float], Tuple[float, float]],
              duration: float = 0.5, dt: float = 2.0e-5) -> BumpResult:
    """Simulate a bump and return its worst instant.

    "Worst" is the peak CONTACT-PATCH load, and the unsprung acceleration is
    sampled at that same index rather than at its own maximum -- they do not
    coincide, and pairing each quantity's own peak would describe an instant
    that never happened.
    """
    res = car.simulate(road, duration=duration, dt=dt)
    i = int(np.argmax(res.Fz_contact))
    a_u = float(res.a_unsprung[i])
    Fz = float(res.Fz_contact[i])
    link = Fz - car.unsprung_mass * (a_u + G)

    return BumpResult(
        Fz_peak=Fz,
        a_unsprung_at_peak=a_u,
        t_peak=float(res.t[i]),
        load_factor=Fz / car.static_wheel_load,
        link_borne_force=link,
        max_wheel_travel=float(np.max(np.abs(res.wheel_travel))),
        wheel_lifted=res.wheel_lifted,
        result=res,
    )


def bump_envelope(car: QuarterCar, height_m: float, length_m: float,
                  speeds, duration: float = 0.5, dt: float = 2.0e-5):
    """One `BumpResult` per traversal speed. Take the worst of what you care about.

    >>> A SINGLE SPEED CANNOT BE THE CONSERVATIVE BUMP CASE, AND NEITHER CAN
        A SINGLE INSTANT. <<<

    Measured on NFR26, 25 mm over 300 mm. The last two columns are the SAME
    quantity -- the force reaching the links -- read at two different moments:

        speed    contact-patch Fz     at the patch peak    max over the trace
         5 m/s      2317 N (3.4x)          1735 N               1852 N
        11 m/s      3206 N (4.8x)          1964 N               2309 N
        20 m/s      4204 N (6.2x)          1838 N               2496 N
        30 m/s      5317 N (7.9x)          1639 N               2552 N
        40 m/s      6486 N (9.6x)          1469 N               2570 N

    Contact-patch load rises monotonically with speed and never turns over: it
    is set by how fast the tire is deflected, so faster is always worse. That
    governs the TIRE, WHEEL, BEARINGS and UPRIGHT.

    The force reaching the LINKS also rises with speed, but far more slowly --
    it saturates, going up 39% across the sweep where the patch load goes up
    180%, because a sharper input is increasingly absorbed by unsprung inertia
    rather than transmitted through the suspension. Its share of the patch load
    falls from 80% to 40%. That saturation is real and it is what the sweep is
    for.

    WHAT THIS DOCSTRING USED TO CLAIM, AND WHY IT WAS WRONG
    -------------------------------------------------------
    It said the link force PEAKS near 11 m/s and then FALLS, so "conservative =
    pick the highest speed" under-sized the pushrod. That reads straight off
    the middle column -- and the middle column is sampled at the wrong instant.
    Read at its own instant (right-hand column) the link force does not turn
    over at all. `force_matrix._bump_result` selected its design speed on the
    middle column and therefore chose 11 m/s, where the linkage is in fact
    least loaded of the fast half of the sweep; the resulting structural
    envelope was 22-27% light on the rod, rocker and upright group.

    So sweep speed AND time, and take the max PER MEMBER -- the worst instant
    differs from member to member, which is why no single `BumpResult` can
    stand in for the trace:

        results = bump_envelope(car, 0.025, 0.300, [4, 8, 11, 15, 20, 25, 30])
        # then walk r.result.Fz_contact / r.result.a_unsprung, NOT r.to_corner_loads(),
        # which pins you to the contact-patch instant. See
        # suspension.force_matrix._swept_bump_envelope.
    """
    return [bump_case(car, half_sine_bump(height_m, length_m, float(v)),
                      duration=duration, dt=dt)
            for v in speeds]
