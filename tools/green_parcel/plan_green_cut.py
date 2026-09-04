"""Per-episode green-segment start, chosen to make the 50 start poses agree.

For each episode we search the legal window [red_release, green_grasp - MIN_APPROACH]
for the instant whose arm pose is closest to a shared reference pose, subject to
the green parcel being visible. The reference is then re-estimated from the
chosen poses and the search repeated, so the reference and the cuts converge
together instead of the reference being guessed up front.
"""
import glob, cv2, json, numpy as np, pyarrow.parquet as pq

R = "/home/zwwl_user2/red_parcel_raw/lerobot_dataset"
OPEN_T, CLOSE_T, CONFIRM, GRIP, J = 0.15, 0.85, 5, 7, 7
LO, HI = np.array([35, 90, 90]), np.array([70, 255, 255])
TAU, THR = 2 * np.pi, 0.004
MIN_APPROACH = 6.0      # seconds of runway that must remain before the grasp
STEP = 0.5


def cycles(g, ts):
    out, rc, ro, st, tc = [], 0, 0, "open", None
    for i, v in enumerate(g):
        rc = rc + 1 if v >= CLOSE_T else 0
        ro = ro + 1 if v <= OPEN_T else 0
        if st == "open" and rc >= CONFIRM:
            st, tc = "closed", float(ts[i])
        elif st == "closed" and ro >= CONFIRM:
            out.append((tc, float(ts[i]))); st, tc = "open", None
    return out


EPS = []
for f in sorted(glob.glob(R + "/data/**/*.parquet", recursive=True)):
    ep = int(f.split("file-")[1][:3])
    t = pq.read_table(f).to_pydict()
    ts = np.asarray(t["timestamp"], dtype=float).ravel()
    act = np.asarray([np.asarray(a) for a in t["action"]], dtype=float)
    st = np.asarray([np.asarray(a) for a in t["observation.state"]], dtype=float).copy()
    sh = st[:, 4].mean() > 1.0
    if sh:
        st[:, 4] -= TAU
    idx = np.arange(0, len(ts), 2)
    cy = cycles(act[idx, GRIP], ts[idx])
    ro, gc, go = cy[0][1], cy[1][0], cy[1][1]
    hi = gc - MIN_APPROACH
    if hi <= ro:                      # gap too short for the runway; take the widest legal span
        hi = ro + 0.5 * (gc - ro)
    grid = np.arange(ro, hi + 1e-6, STEP)
    if len(grid) == 0:
        grid = np.array([ro])
    poses = np.array([st[min(int(np.searchsorted(ts, g)), len(ts) - 1)][:J] for g in grid])
    EPS.append(dict(ep=ep, ts=ts, ro=ro, gc=gc, go=go, end=float(ts[-1]),
                    grid=grid, poses=poses, j5=sh))
print("episodes:", len(EPS), " 窗口点数 min/med/max:",
      min(len(e["grid"]) for e in EPS), int(np.median([len(e["grid"]) for e in EPS])),
      max(len(e["grid"]) for e in EPS))

# ---- green fraction on the same grid, one decode pass per episode/camera ----
for e in EPS:
    for cam in ("chest", "wrist_right"):
        cap = cv2.VideoCapture(f"{R}/videos/observation.images.{cam}/chunk-000/file-{e['ep']:03d}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        v = []
        for g in e["grid"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(0.0, g) * fps)))
            ok, img = cap.read()
            if not ok:
                v.append(0.0); continue
            hsv = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2HSV)
            v.append(float(cv2.inRange(hsv, LO, HI).mean() / 255.0))
        cap.release()
        e[cam] = np.array(v)
print("视频网格缓存完成")

# ---- alternate: reference pose <-> per-episode choice ----
ref = np.median(np.array([e["poses"][len(e["poses"]) // 2] for e in EPS]), axis=0)
for it in range(8):
    sel = []
    for e in EPS:
        d = np.linalg.norm(e["poses"] - ref, axis=1)
        pen = np.where(e["chest"] > THR, 0.0, 3.0) + np.where(e["wrist_right"] > THR, 0.0, 0.6)
        k = int(np.argmin(d + pen))
        sel.append(k)
    P = np.array([e["poses"][k] for e, k in zip(EPS, sel)])
    new = np.median(P, axis=0)
    shift = float(np.linalg.norm(new - ref))
    ref = new
    if shift < 1e-4:
        break
print("参考位姿迭代 %d 轮收敛, 最后位移 %.2e" % (it + 1, shift))

spread = P.max(0) - P.min(0)
ch = np.array([e["chest"][k] for e, k in zip(EPS, sel)])
wr = np.array([e["wrist_right"][k] for e, k in zip(EPS, sel)])
t0 = np.array([e["grid"][k] for e, k in zip(EPS, sel)])
marg = np.array([e["gc"] - t for e, t in zip(EPS, t0)])
dur = np.array([e["end"] - t for e, t in zip(EPS, t0)])
d2ref = np.linalg.norm(P - ref, axis=1)

print()
print("参考位姿 J1..J7:", np.round(ref, 3).tolist())
print("起始位姿极差    :", np.round(spread, 3).tolist())
print("  最宽 %.3f rad = %.1f deg   (固定 红释放+8s 时是 108.4 deg)" % (spread.max(), np.degrees(spread.max())))
print("到参考的距离    : med %.3f  p90 %.3f  max %.3f rad" % (np.median(d2ref), np.percentile(d2ref, 90), d2ref.max()))
print("胸部可见 %d/50   腕部可见 %d/50" % (int((ch > THR).sum()), int((wr > THR).sum())))
print("起点 t0         : min %.1f med %.1f max %.1f s" % (t0.min(), np.median(t0), t0.max()))
print("距绿抓取余量    : min %.1f p10 %.1f med %.1f max %.1f s" % (marg.min(), np.percentile(marg, 10), np.median(marg), marg.max()))
print("段时长          : min %.1f med %.1f max %.1f s" % (dur.min(), np.median(dur), dur.max()))
print("总帧数 ~ %d" % int(sum(dur) * 30))

plan = []
for e, k, dd in zip(EPS, sel, d2ref):
    plan.append(dict(ep=e["ep"], t0=float(e["grid"][k]), t1=float(e["end"]),
                     dur=float(e["end"] - e["grid"][k]), margin=float(e["gc"] - e["grid"][k]),
                     chest=float(e["chest"][k]), wrist=float(e["wrist_right"][k]),
                     dist_to_ref=float(dd), j5_shifted=bool(e["j5"]),
                     t_red_open=float(e["ro"]), t_green_close=float(e["gc"]), t_green_open=float(e["go"]),
                     pose=[float(x) for x in e["poses"][k]]))
json.dump(dict(reference_pose=[float(x) for x in ref], min_approach_s=MIN_APPROACH,
               step_s=STEP, episodes=plan), open("/tmp/green_adaptive_plan.json", "w"), indent=1)
print("写出 /tmp/green_adaptive_plan.json")

print()
print("到参考最远的 6 条(需人工看):")
for i in np.argsort(-d2ref)[:6]:
    p = plan[i]
    print("  ep%d  t0=%.1f  距参考 %.2f rad  余量 %.1fs  胸 %.2f%%  腕 %.2f%%"
          % (p["ep"], p["t0"], p["dist_to_ref"], p["margin"], p["chest"] * 100, p["wrist"] * 100))


def sheet(moment, cam, fname):
    tiles = []
    for e, k in zip(EPS, sel):
        tt = e["grid"][k] if moment == "start" else max(0.0, e["end"] - 0.2)
        cap = cv2.VideoCapture(f"{R}/videos/observation.images.{cam}/chunk-000/file-{e['ep']:03d}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tt * fps)))
        ok, img = cap.read(); cap.release()
        img = cv2.resize(img, (256, 144)) if ok else np.zeros((144, 256, 3), np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        g = float(cv2.inRange(hsv, LO, HI).mean() / 255.0)
        cv2.rectangle(img, (0, 0), (256, 16), (0, 0, 0), -1)
        col = (0, 255, 0) if g > THR else (0, 165, 255)
        cv2.putText(img, "ep%d t=%.1f g=%.2f%%" % (e["ep"], tt, g * 100), (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 10:
        tiles.append(np.zeros((144, 256, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
    cv2.imwrite(fname, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", fname)


for cam in ("chest", "wrist_right"):
    sheet("start", cam, f"/tmp/adap_start_{cam}.jpg")
