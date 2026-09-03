# Release Notes — v3.0

## Status
Stable **research-core** release.

## What is frozen
- Direction + scale + neighbor semantic primitive.
- Shared octree/topology-signature compiler cache.
- `SUBFACE_SPARSE` CPU representation.
- `MODAL_STREAM` compact-state representation.
- Compatible cell-pressure / face-flux projection.
- Conservative planner policy with 5% runtime tie band.

## Release decision
v3.0 is a stabilization milestone, not proof of production readiness.

## Negative results retained
- Early runtime abstraction overhead.
- Initial incompatible projection.
- v2.9 energy collapse.
- v2.11 face/cell low-pass failure.
- v2.17 mean-only modal accuracy loss.
- v2.20 global modal-matrix scaling failure at 64³.
- Long-time error accumulation in v2.21.
- Lack of robust modal CPU speed advantage in v2.23.
