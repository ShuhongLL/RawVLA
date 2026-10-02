"""Runtime LIBERO / MuJoCo lighting controls for RAWVLA-Bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import numpy as np

MAX_LIGHT_INTENSITY = 10000.0
COMMON_AGENTVIEW_KEY_RIG = "common_agentview_key"
COMMON_AGENTVIEW_KEY_BASE_INTENSITY = 0.02
LIGHTING_EV_DISTRIBUTION = "uniform"
LIGHTING_DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
SENSOR_SATURATION_FORMULA = (
    "sensor_saturation_ev = interp(sensor_saturation_ev_range, lighting_sample_unit) "
    "when configured, else max(lighting_ev - 1.5, 0.0) * 0.12"
)
AGENTVIEW_POS = np.asarray([0.65861, 0.0, 1.61035], dtype=np.float32)
TABLE_FLOOD_TARGETS = (
    (-0.45, -0.55, 0.75),
    (-0.45, 0.0, 0.75),
    (-0.45, 0.55, 0.75),
    (0.0, -0.55, 0.75),
    (0.0, 0.0, 0.75),
    (0.0, 0.55, 0.75),
    (0.45, -0.55, 0.75),
    (0.45, 0.0, 0.75),
    (0.45, 0.55, 0.75),
)


@dataclass(frozen=True)
class LightingDomainConfig:
    ev_range: tuple[float, float]
    lighting_rig: str = COMMON_AGENTVIEW_KEY_RIG
    agentview_table_flood: float = 0.0
    agentview_table_flood_range: tuple[float, float] | None = None
    sensor_saturation_ev_range: tuple[float, float] | None = None
    sensor_saturation_start_ev: float = 1.5
    sensor_saturation_slope: float = 0.12
    description: str = ""

    def sensor_saturation_ev(self, lighting_ev: float) -> float:
        return float(max(float(lighting_ev) - self.sensor_saturation_start_ev, 0.0) * self.sensor_saturation_slope)

    def sensor_saturation_for_unit_sample(self, lighting_ev: float, sample_unit: float | None = None) -> float:
        if self.sensor_saturation_ev_range is None:
            return self.sensor_saturation_ev(lighting_ev)
        if sample_unit is None:
            raise ValueError("sample_unit is required for domains with sensor_saturation_ev_range")
        low, high = self.sensor_saturation_ev_range
        return float(low + float(sample_unit) * (high - low))

    def table_flood_for_unit_sample(self, sample_unit: float | None = None) -> float:
        if self.agentview_table_flood_range is None:
            return float(self.agentview_table_flood)
        if sample_unit is None:
            raise ValueError("sample_unit is required for domains with agentview_table_flood_range")
        low, high = self.agentview_table_flood_range
        return float(low + float(sample_unit) * (high - low))


LIGHTING_DOMAIN_CONFIGS: dict[str, LightingDomainConfig] = {
    "ExtremeLow": LightingDomainConfig(
        ev_range=(-5.5, -4.0),
        description="severe low light; signal close to RAW noise floor",
    ),
    "Low": LightingDomainConfig(
        ev_range=(-4.0, -2.5),
        description="low light; recoverable semantics with visible RAW noise",
    ),
    "Normal": LightingDomainConfig(
        ev_range=(-0.5, 0.5),
        description="near nominal LIBERO lighting",
    ),
    "Over": LightingDomainConfig(
        ev_range=(1.5, 3.0),
        agentview_table_flood=0.5,
        agentview_table_flood_range=(0.0, 1.0),
        sensor_saturation_ev_range=(0.5, 1.0),
        description="strong illumination with partial highlights; scene remains recoverable",
    ),
    "ExtremeOver": LightingDomainConfig(
        ev_range=(6.0, 8.0),
        agentview_table_flood=1.5,
        agentview_table_flood_range=(1.0, 2.0),
        sensor_saturation_ev_range=(1.0, 1.5),
        description="extra agentview table flood plus RAW full-well pressure for strong but recoverable clipping",
    ),
}
DEFAULT_EV_RANGES: dict[str, tuple[float, float]] = {
    domain: config.ev_range for domain, config in LIGHTING_DOMAIN_CONFIGS.items()
}


def get_lighting_domain_config(domain: str) -> LightingDomainConfig:
    try:
        return LIGHTING_DOMAIN_CONFIGS[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown RAWVLA-Bench lighting domain: {domain!r}") from exc


def sensor_saturation_ev_for_lighting(lighting_ev: float, domain: str | None = None) -> float:
    config = get_lighting_domain_config(domain) if domain is not None else LIGHTING_DOMAIN_CONFIGS["Over"]
    return config.sensor_saturation_ev(lighting_ev)


def lighting_parameters_for_sample(
    domain: str,
    lighting_ev: float,
    *,
    sample_unit: float | None = None,
) -> dict[str, float | str]:
    config = get_lighting_domain_config(domain)
    return {
        "lighting_ev": float(lighting_ev),
        "lighting_scale": float(2.0 ** float(lighting_ev)),
        "lighting_rig": config.lighting_rig,
        "agentview_table_flood": config.table_flood_for_unit_sample(sample_unit),
        "sensor_saturation_ev": config.sensor_saturation_for_unit_sample(lighting_ev, sample_unit),
    }


def lighting_domain_configs_for_manifest(
    ev_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {}
    for domain in LIGHTING_DOMAINS:
        config = asdict(get_lighting_domain_config(domain))
        config["ev_range"] = list(ev_ranges[domain] if ev_ranges is not None else config["ev_range"])
        config["lighting_distribution"] = LIGHTING_EV_DISTRIBUTION
        config["sensor_saturation_formula"] = SENSOR_SATURATION_FORMULA
        configs[domain] = config
    return configs


@dataclass(frozen=True)
class LightingState:
    light_ambient: np.ndarray | None
    light_diffuse: np.ndarray | None
    light_specular: np.ndarray | None
    headlight_ambient: np.ndarray | None
    headlight_diffuse: np.ndarray | None
    headlight_specular: np.ndarray | None


def _copy_attr(obj: object, name: str) -> np.ndarray | None:
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    if value is None:
        return None
    return np.array(value, dtype=np.float32, copy=True)


def capture_lighting_state(env: object) -> LightingState:
    model = env.sim.model
    headlight = getattr(getattr(model, "vis", None), "headlight", None)
    return LightingState(
        light_ambient=_copy_attr(model, "light_ambient"),
        light_diffuse=_copy_attr(model, "light_diffuse"),
        light_specular=_copy_attr(model, "light_specular"),
        headlight_ambient=_copy_attr(headlight, "ambient") if headlight is not None else None,
        headlight_diffuse=_copy_attr(headlight, "diffuse") if headlight is not None else None,
        headlight_specular=_copy_attr(headlight, "specular") if headlight is not None else None,
    )


def _capture_original_xml(env: object) -> str | None:
    sim_model = getattr(getattr(env, "sim", None), "model", None)
    if sim_model is not None and hasattr(sim_model, "get_xml"):
        return str(sim_model.get_xml())
    model = getattr(getattr(env, "env", None), "model", None)
    if model is None or not hasattr(model, "get_xml"):
        return None
    return str(model.get_xml())


def _with_common_agentview_key_rig(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML has no worldbody; cannot add RAWVLA-Bench lights")
    existing_light_names = {
        light.get("name")
        for light in worldbody.findall("light")
        if light.get("name") is not None
    }
    expected_light_names = {"rawvla_agentview_key"} | {
        f"rawvla_agentview_flood_{idx:02d}" for idx in range(len(TABLE_FLOOD_TARGETS))
    }
    if expected_light_names.issubset(existing_light_names):
        return ET.tostring(root, encoding="unicode")

    color = (
        f"{COMMON_AGENTVIEW_KEY_BASE_INTENSITY:g} "
        f"{COMMON_AGENTVIEW_KEY_BASE_INTENSITY:g} "
        f"{COMMON_AGENTVIEW_KEY_BASE_INTENSITY:g}"
    )
    # Approximate LIBERO's desktop / agentview camera pose. The light is a
    # fixed camera-side key light aimed at the tabletop workspace, so all five
    # domains share the same illumination topology and differ only in strength.
    if "rawvla_agentview_key" not in existing_light_names:
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "rawvla_agentview_key",
                "pos": "0.65861 0 1.61035",
                "dir": "-0.63076 0 -0.77598",
                "directional": "false",
                "cutoff": "90",
                "exponent": "1",
                "diffuse": color,
                "specular": color,
                "ambient": color,
                "castshadow": "false",
            },
        )
    for idx, target in enumerate(TABLE_FLOOD_TARGETS):
        if f"rawvla_agentview_flood_{idx:02d}" in existing_light_names:
            continue
        direction = np.asarray(target, dtype=np.float32) - AGENTVIEW_POS
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": f"rawvla_agentview_flood_{idx:02d}",
                "pos": "0.65861 0 1.61035",
                "dir": " ".join(f"{float(v):.5f}" for v in direction),
                "directional": "false",
                "cutoff": "90",
                "exponent": "1",
                "diffuse": "0 0 0",
                "specular": "0 0 0",
                "ambient": "0 0 0",
                "castshadow": "false",
            },
        )
    return ET.tostring(root, encoding="unicode")


def rawvla_light_xml_from_env(env: object) -> str:
    """Return the current compiled scene XML with RAWVLA-Bench light slots.

    Call this after the normal LIBERO episode reset for the target seed. Some
    LIBERO suites encode fixed fixture placement in the compiled XML rather
    than in qpos, so generating the RAW XML before reset can produce a scene
    that no longer matches the demonstration init state.
    """
    original_xml = _capture_original_xml(env)
    if original_xml is None:
        raise RuntimeError("Cannot build RAWVLA-Bench light XML: env model does not expose get_xml().")
    return _with_common_agentview_key_rig(original_xml)


def rawvla_light_xml_from_string(xml_string: str) -> str:
    """Return an existing MuJoCo XML string with RAWVLA-Bench light slots."""
    return _with_common_agentview_key_rig(str(xml_string))


def _has_light(env: object, light_name: str) -> bool:
    try:
        env.sim.model.light_name2id(light_name)
    except Exception:
        return False
    return True


def _has_rawvla_light_slots(env: object) -> bool:
    return _has_light(env, "rawvla_agentview_key") and all(
        _has_light(env, f"rawvla_agentview_flood_{idx:02d}") for idx in range(len(TABLE_FLOOD_TARGETS))
    )


def install_rawvla_light_slots_once(env: object) -> None:
    """Compile RAWVLA-Bench's real extra lights into the MuJoCo model once.

    Prefer ``rawvla_light_xml_from_env`` after the normal episode reset so the
    RAW XML inherits suite-specific fixture placement from the compiled scene.
    Per-domain lighting changes must only update MuJoCo light arrays; they must
    not recompile XML after ``set_init_state``.
    """
    if getattr(env, "_rawvla_bench_light_slots_installed", False) and _has_rawvla_light_slots(env):
        return

    original_xml = _capture_original_xml(env)
    if original_xml is None:
        raise RuntimeError("Cannot install RAWVLA-Bench lights: env model does not expose get_xml().")
    if not _has_rawvla_light_slots(env):
        env.reset_from_xml_string(_with_common_agentview_key_rig(original_xml))

    setattr(env, "_rawvla_bench_light_slots_installed", True)
    if hasattr(env, "_rawvla_bench_lighting_baseline"):
        delattr(env, "_rawvla_bench_lighting_baseline")


def prepare_lighting_rig(env: object, lighting_rig: str = "default") -> None:
    if lighting_rig == COMMON_AGENTVIEW_KEY_RIG:
        install_rawvla_light_slots_once(env)


def _assign_scaled(obj: object, name: str, baseline: np.ndarray | None, scale: float) -> None:
    if baseline is None or not hasattr(obj, name):
        return
    getattr(obj, name)[:] = np.clip(baseline * np.float32(scale), 0.0, MAX_LIGHT_INTENSITY)


def _set_named_light_intensity(env: object, light_name: str, intensity: float, *, require: bool = False) -> None:
    try:
        light_id = env.sim.model.light_name2id(light_name)
    except Exception as exc:
        if require:
            raise RuntimeError(f"RAWVLA-Bench light slot is not installed: {light_name}") from exc
        return
    clipped = float(np.clip(float(intensity), 0.0, MAX_LIGHT_INTENSITY))
    value = np.asarray([clipped, clipped, clipped], dtype=np.float32)
    env.sim.model.light_ambient[light_id, :] = value
    env.sim.model.light_diffuse[light_id, :] = value
    env.sim.model.light_specular[light_id, :] = value


def _set_agentview_table_flood(env: object, intensity: float, *, require: bool = False) -> None:
    for idx in range(len(TABLE_FLOOD_TARGETS)):
        _set_named_light_intensity(env, f"rawvla_agentview_flood_{idx:02d}", intensity, require=require)


def apply_lighting_ev(
    env: object,
    lighting_ev: float,
    lighting_rig: str = "default",
    agentview_table_flood: float = 0.0,
) -> float:
    """Scale the simulator's original light intensities by ``2 ** lighting_ev``.

    The first call captures the environment's original light and headlight
    arrays. Later calls always scale from that baseline, not from the previous
    condition, so the five domains can be rendered in any order.
    """
    uses_rawvla_slots = lighting_rig == COMMON_AGENTVIEW_KEY_RIG
    if uses_rawvla_slots and not _has_rawvla_light_slots(env):
        raise RuntimeError(
            "RAWVLA-Bench common_agentview_key requires real XML light slots. "
            "Reset the episode normally, build rawvla_light_xml_from_env(env), "
            "then reset from that RAW XML before set_init_state()."
        )

    if not hasattr(env, "_rawvla_bench_lighting_baseline"):
        setattr(env, "_rawvla_bench_lighting_baseline", capture_lighting_state(env))

    state: LightingState = getattr(env, "_rawvla_bench_lighting_baseline")
    scale = float(2.0 ** float(lighting_ev))
    model = env.sim.model
    _assign_scaled(model, "light_ambient", state.light_ambient, scale)
    _assign_scaled(model, "light_diffuse", state.light_diffuse, scale)
    _assign_scaled(model, "light_specular", state.light_specular, scale)

    headlight = getattr(getattr(model, "vis", None), "headlight", None)
    if headlight is not None:
        _assign_scaled(headlight, "ambient", state.headlight_ambient, scale)
        _assign_scaled(headlight, "diffuse", state.headlight_diffuse, scale)
        _assign_scaled(headlight, "specular", state.headlight_specular, scale)

    _set_agentview_table_flood(env, float(agentview_table_flood), require=uses_rawvla_slots)
    env.sim.forward()
    return scale
