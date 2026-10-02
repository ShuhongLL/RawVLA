# Release checklist

This directory is a clean publication snapshot. Before making it public:

- [x] Add the root MIT `LICENSE` (third-party submodules keep their own licenses).
- [ ] Push each project-maintained submodule branch before pushing `main`.
- [ ] Confirm every pinned submodule commit is anonymously fetchable.
- [ ] Run `python scripts/check_release.py --strict` in a fully initialized checkout.
- [ ] Run the RAW-VLA smoke test in the StarVLA environment.
- [ ] Clone recursively into a new directory and execute one small LIBERO replay.
- [ ] Execute one small RoboTwin replay/evaluation on a simulator-capable machine.

Project-maintained submodule branches can be pushed in this order:

```bash
git -C starVLA push origin submodules/starVLA
git -C third_party/RoboTwin push origin submodules/RoboTwin
git -C third_party/FastWAM push origin submodules/FastWAM
```

Then publish the root snapshot:

```bash
git push -u origin main
```

The local staging clones use Git alternates to save disk space. They are valid
for preparing and pushing this release, but the portability test must be done
from a fresh recursive clone after the referenced commits are on the remote.
