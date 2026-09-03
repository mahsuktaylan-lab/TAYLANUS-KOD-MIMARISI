#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys, time, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("taylanus_core",ROOT/"taylanus_core.py")
t=importlib.util.module_from_spec(spec); sys.modules["taylanus_core"]=t; spec.loader.exec_module(t)
expected=json.loads((ROOT/"benchmarks"/"expected_metrics.json").read_text())

for case in expected["cases"]:
    NF=int(case["NF"]); rep=case["representation"]
    dt=float(case["dt"]); tf=float(case["t_final"])
    t.CURRENT_NF,t.CURRENT_MAX_LEVEL,t.CURRENT_HF,t.CURRENT_FACE_AREA=t.make_context(NF)
    keys=t.build_initial_topology(NF)
    ir=t.UnifiedTaylanusIR(NF,keys,t.UnifiedCompileCache())
    face0=ir.make_initial_subface(dt)
    steps=int(round(tf/dt))
    s=time.perf_counter()
    if rep=="SUBFACE_SPARSE":
        state=face0
        for _ in range(steps): state=ir.step_subface(state,dt)
    else:
        state=ir.subface_to_modal(face0,dt)
        for _ in range(steps): state=ir.step_modal(state,dt)
    rt=time.perf_counter()-s
    print(f"NF={NF:2d} {rep:15s} steps={steps:4d} runtime={rt:.6f}s")
