# Submodule release map

RAW-VLA uses submodules so upstream projects and project-specific adaptations
remain auditable. Project-maintained submodules are hosted as branches of the
same public repository; upstream-only dependencies retain their original URLs.

| Path | Branch | Purpose |
|---|---|---|
| `starVLA` | `submodules/starVLA` | RAW frontend registry, training, serving, LIBERO/RoboTwin adapters |
| `third_party/LIBERO` | `submodules/LIBERO` | LIBERO simulator integration |
| `third_party/openvla-oft` | `submodules/openvla-oft` | OpenVLA-OFT dependency |
| `third_party/RoboTwin` | `submodules/RoboTwin` | RAW camera and policy deployment integration |
| `third_party/Isaac-GR00T` | upstream `main` | GR00T dependency |
| `third_party/FastWAM` | `submodules/FastWAM` | FastWAM/RoboTwin integration, based on upstream FastWAM |

## Publish order

1. Push project-maintained submodule commits/branches first.
2. Ensure every commit referenced by the root gitlinks is fetchable without
   authentication.
3. Push the root `main` branch.
4. Test a fresh anonymous `git clone --recurse-submodules` in a new directory.

The release checker reports a missing or mismatched checkout, but only a fresh
network clone can prove that every referenced commit is publicly reachable.
