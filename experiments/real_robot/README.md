# Real-Robot Integrations

Real-robot deployment is intentionally separated from simulator benchmarks and
synthetic ISP perturbation experiments.

The maintained robot-specific implementations currently live in the StarVLA
submodule under [`starVLA/examples/realRobots`](../../starVLA/examples/realRobots).
They retain their upstream directory layout so StarVLA imports and configuration
resolution continue to work. Root-owned launchers or calibration files for a
specific physical platform should be added under this directory instead of the
repository root.

RoboTwin and LIBERO are simulators and do not belong in this directory. Replay
and trajectory-generation utilities remain under `replay/` and `scripts/`.
