"""Detection metric for cerebral microbleeds: connected components, 26-connectivity.

Per case:  TP = reference components touched by >=1 predicted component
           FN = reference components that are not touched
           FP = predicted components touching no reference component
           (several predictions on one reference count as 1 TP and 0 FP)
Case level: positive = at least one component -> TP/FP/TN/FN per case
           (true negatives exist only here).
Overall:   pooled over all cases (sums), F1 = 2TP/(2TP+FP+FN), plus false
           positives per case and the mean counting error |n_pred - n_ref|.

Library:  from metric import hits;  hits(ref_bool, pred_bool) -> dict
Usage:    python metric.py --net <dir> --mod t2star [--cases cases.json]
          expects <dir>/fold<k>/<id>_pred.nii.gz (binary, on the image grid), every
          case exactly once (in its test fold); writes <dir>/results.csv (per case,
          stays local) and <dir>/summary.json (sums only, publishable).
          python metric.py --selftest    checks the logic on synthetic masks.
          python metric.py --crosscheck <path/to/mb_metrik.py>
                                         proves this file behaves identically to the
                                         German original the other cohort was scored
                                         with -- see the note at the bottom.

This is a translation of mb_metrik.py (identical algorithm, English names and keys).
The translation is not taken on trust: --crosscheck runs both implementations over
random volumes and fails on the first disagreement."""
import os, sys, glob, json, argparse
import numpy as np
from scipy import ndimage

N26 = np.ones((3, 3, 3), dtype=bool)

def hits(ref, pred):
    ref = np.asarray(ref) > 0; pred = np.asarray(pred) > 0
    lr, nr = ndimage.label(ref, structure=N26)
    lp, npred = ndimage.label(pred, structure=N26)
    # touching = voxel overlap of the components
    ref_hit = np.zeros(nr + 1, bool); pred_hit = np.zeros(npred + 1, bool)
    both = ref & pred
    if both.any():
        ref_hit[np.unique(lr[both])] = True
        pred_hit[np.unique(lp[both])] = True
    tp = int(ref_hit[1:].sum()); fn = nr - tp; fp = int((~pred_hit[1:]).sum())
    return dict(n_ref=int(nr), n_pred=int(npred), tp=tp, fp=fp, fn=fn,
                case_ref=int(nr > 0), case_pred=int(npred > 0), count_error=abs(int(npred) - int(nr)))

def summarize(rows):
    s = {k: int(sum(z[k] for z in rows)) for k in ("n_ref", "n_pred", "tp", "fp", "fn")}
    n = len(rows)
    p = s["tp"] / max(s["tp"] + s["fp"], 1); r = s["tp"] / max(s["tp"] + s["fn"], 1)
    case = dict(tp=sum(z["case_ref"] and z["case_pred"] for z in rows), fp=sum((not z["case_ref"]) and z["case_pred"] for z in rows),
                tn=sum((not z["case_ref"]) and (not z["case_pred"]) for z in rows), fn=sum(z["case_ref"] and (not z["case_pred"]) for z in rows))
    return dict(cases=n, **s, precision=round(p, 4), sensitivity=round(r, 4),
                f1=round(2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1), 4),
                fp_per_case=round(s["fp"] / max(n, 1), 3), count_error_mean=round(float(np.mean([z["count_error"] for z in rows])), 3),
                case_level={k: int(v) for k, v in case.items()},
                case_accuracy=round((case["tp"] + case["tn"]) / max(n, 1), 4))

def selftest():
    def ball(a, c, r):
        z, y, x = np.ogrid[:a.shape[0], :a.shape[1], :a.shape[2]]
        a[((z-c[0])**2 + (y-c[1])**2 + (x-c[2])**2) <= r*r] = 1
    ref = np.zeros((40, 40, 40), np.uint8); pred = np.zeros_like(ref)
    ball(ref, (10, 10, 10), 3); ball(ref, (25, 25, 25), 2); ball(ref, (30, 8, 30), 2)   # 3 references
    ball(pred, (10, 10, 10), 2); ball(pred, (11, 13, 10), 1)   # two predictions on reference 1 -> 1 TP, 0 FP
    ball(pred, (25, 25, 25), 3)                                # reference 2 hit
    pred[5, 35, 5] = 1; pred[6, 36, 6] = 1                     # diagonally connected only: ONE FP under 26-connectivity
    t = hits(ref, pred)
    want = dict(n_ref=3, n_pred=3, tp=2, fp=1, fn=1, case_ref=1, case_pred=1, count_error=0)
    assert t == want, (t, want)
    empty = hits(np.zeros((5, 5, 5)), np.zeros((5, 5, 5)))
    assert empty["case_ref"] == 0 and empty["case_pred"] == 0 and empty["tp"] == empty["fp"] == empty["fn"] == 0
    z = summarize([t, empty, hits(np.zeros((5, 5, 5)), pred[:5, :5, :5] * 0 + 1)])
    assert z["tp"] == 2 and z["fp"] == 2 and z["fn"] == 1 and z["case_level"] == dict(tp=1, fp=1, tn=1, fn=0), z
    assert abs(z["f1"] - 4 / 7) < 1e-3, z["f1"]
    print("selftest passed:", t, "| overall:", z)

def crosscheck(pfad, n=200, seed=0):
    """Run the German original and this translation over random volumes and compare.

    Why this exists: the other cohort was scored with mb_metrik.py. If the numbers of
    the two projects are to be compared, the translation must not merely look right."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mb_metrik_orig", pfad)
    orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
    umbenannt = dict(fall_ref="case_ref", fall_pred="case_pred", zaehlfehler="count_error")
    rng = np.random.default_rng(seed)
    for i in range(n):
        shape = tuple(int(x) for x in rng.integers(6, 20, 3))
        ref = (rng.random(shape) < rng.choice([0.0, 0.02, 0.1])).astype(np.uint8)
        pred = (rng.random(shape) < rng.choice([0.0, 0.02, 0.1])).astype(np.uint8)
        a = orig.treffer(ref, pred); b = hits(ref, pred)
        a = {umbenannt.get(k, k): v for k, v in a.items()}
        assert a == b, f"Lauf {i}, shape {shape}: Original {a} != Uebersetzung {b}"
    print(f"crosscheck passed: {n} random volumes, identical results to {os.path.basename(pfad)}")

def evaluate(net, mod, cases_path):
    import nibabel as nib, csv
    cases = json.load(open(cases_path))[mod]["cases"]
    rows, missing, duplicate = [], [], []
    for f in cases:
        preds = sorted(glob.glob(f"{net}/fold*/{f['id']}_pred.nii.gz"))
        if not preds: missing.append(f["id"]); continue
        if len(preds) > 1: duplicate.append(f["id"])
        p = preds[0]; expected = f"{net}/fold{f['fold']}/{f['id']}_pred.nii.gz"
        if p != expected: duplicate.append(f["id"] + " (wrong fold)")
        ir, ip = nib.load(f["mask"]), nib.load(p)
        if ir.shape[:3] != ip.shape[:3]:
            raise SystemExit(f"{f['id']}: prediction {ip.shape[:3]} not on image grid {ir.shape[:3]}")
        t = hits(np.asarray(ir.dataobj), np.asarray(ip.dataobj))
        rows.append(dict(id=f["id"], fold=f["fold"], klass=f["klass"], **t))
    if missing or duplicate:
        print(f"WARNING: {len(missing)} cases without prediction, {len(duplicate)} duplicate/wrong fold:", duplicate[:5])
    if not rows: raise SystemExit("no predictions found")
    with open(f"{net}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    z = summarize(rows); z["net"] = os.path.basename(net.rstrip("/")); z["modality"] = mod
    z["cases_without_prediction"] = len(missing)
    z["per_fold"] = {k: summarize([x for x in rows if x["fold"] == k])["f1"] for k in sorted({x["fold"] for x in rows})}
    json.dump(z, open(f"{net}/summary.json", "w"), indent=1)
    print(json.dumps(z))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--net"); ap.add_argument("--mod", choices=["swi", "t2star"])
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json"))
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--crosscheck")
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.crosscheck: crosscheck(a.crosscheck)
    elif a.net and a.mod: evaluate(a.net, a.mod, a.cases)
    else: ap.print_help()
