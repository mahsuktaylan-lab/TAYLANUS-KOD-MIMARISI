#!/usr/bin/env python3
from pathlib import Path
import json
import importlib.util
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/"taylanus_core.py"

spec=importlib.util.spec_from_file_location("taylanus_core",CORE)
t=importlib.util.module_from_spec(spec)
sys.modules["taylanus_core"]=t
spec.loader.exec_module(t)

expected=json.loads((ROOT/"benchmarks"/"expected_metrics.json").read_text())
refs={
    32:np.load(ROOT/"references"/"localized_vortex_N32_t0p20.npy"),
    64:np.load(ROOT/"references"/"localized_vortex_N64_t0p20.npy"),
}

def run_case(case):
    NF=int(case["NF"]); rep=case["representation"]
    dt=float(case["dt"]); tf=float(case["t_final"])
    t.CURRENT_NF,t.CURRENT_MAX_LEVEL,t.CURRENT_HF,t.CURRENT_FACE_AREA=t.make_context(NF)
    keys=t.build_initial_topology(NF)
    ir=t.UnifiedTaylanusIR(NF,keys,t.UnifiedCompileCache())
    face0=ir.make_initial_subface(dt)
    steps=int(round(tf/dt))

    if rep=="SUBFACE_SPARSE":
        state=face0
        for _ in range(steps):
            state=ir.step_subface(state,dt)
        face=state
        dofs=sum(len(x) for x in state)
        div=t.subface_divergence(face,ir.geom)[0]
    else:
        state=ir.subface_to_modal(face0,dt)
        for _ in range(steps):
            state=ir.step_modal(state,dt)
        face=ir.modal_to_subface(state)
        dofs=sum(len(x) for x in state)
        b=ir.ensure_modal()
        div=t.modal_div(state,ir.geom,b["H"],b["maps"],b["pback"])

    ref=refs[NF]
    l2=t.rel_l2_face(face,ir.geom,ir.sub_ir,ref)
    eref=.5*np.mean(np.sum(ref*ref,axis=-1))
    eerr=abs(t.energy_face(face,ir.geom,ir.sub_ir)-eref)/eref

    l2d=abs(l2-case["L2"])/abs(case["L2"])
    ed=abs(eerr-case["energy_error"])/abs(case["energy_error"])
    ok=(
        l2d < expected["gates"]["relative_L2_delta_max"] and
        ed < expected["gates"]["relative_energy_delta_max"] and
        dofs == int(case["state_dofs"]) and
        div < expected["gates"]["RMS_divergence_max"]
    )
    return NF,rep,l2,eerr,dofs,div,l2d,ed,ok

all_ok=True
for case in expected["cases"]:
    r=run_case(case)
    print(
        f"NF={r[0]:2d} {r[1]:15s} "
        f"L2={r[2]:.8f} Eerr={r[3]:.8f} DOFs={r[4]} "
        f"div={r[5]:.3e} PASS={r[-1]}"
    )
    all_ok &= r[-1]

if not all_ok:
    raise SystemExit(1)
print("TAYLANUS v3.0 numerical regression: PASS")
