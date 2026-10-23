"""Record FR3 Duo handover episodes in the evaluation scene (runs INSIDE Isaac Sim's Python: numpy + Pillow only).

Scene setup, cameras, gripper mapping and the replay timing are those of the team's `eval_isaac.py`
(github.com/yugalrk/Hack), whose replay puts the box on the tray in this setup. The reference file comes from
`hackpipe sim reference` (demo actions + per-episode lighting of our approved dataset version).

Modes
-----
snapshot : the 3 cameras at reset, lit like each training episode → out/snapshots/<episode>/<camera>.png
           (`hackpipe sim camera-gap` compares them with the training frames).
render   : replay the demo actions with each training episode's lighting and record every frame — the training
           data re-rendered by the simulator the policy is evaluated in.
dart     : like render, but the executed arm targets get start-pose noise and Ornstein-Uhlenbeck noise while the
           *recorded* action stays the demo's target (DART). The robot drifts off the demo path and the label says
           how to get back, so the policy learns corrections. Episodes whose box doesn't end on the tray are kept
           with success=false; `hackpipe sim convert` leaves them out by default.

Recording rule (as in the organisers' data and eval_isaac replay): frame k = state at physics tick 2k (60 Hz),
action[k] = target for the next 2 ticks; 30 fps.

Usage (site PC):
    <isaac>/python.sh scripts/isaac/record_demos.py --mode snapshot --usd <scene> --ref ref.json --out rec/snap --limit 5
    <isaac>/python.sh scripts/isaac/record_demos.py --mode render --usd <scene> --ref ref.json --out rec/render
    <isaac>/python.sh scripts/isaac/record_demos.py --mode dart --usd <scene> --ref ref.json --out rec/dart --repeats 2
Then on any machine with hackpipe:  hackpipe sim convert --recordings rec/render --name fr3_duo_handover_sim_render
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["snapshot", "render", "dart"], required=True)
ap.add_argument("--usd", required=True, help="Hackathon_FR3Duo_flattened.usd")
ap.add_argument("--ref", required=True, help="reference JSON from `hackpipe sim reference`")
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="train,val", help="which record_episodes to use (train / val)")
ap.add_argument("--limit", type=int, default=None, help="only the first N episodes")
ap.add_argument("--repeats", type=int, default=1, help="dart: noisy recordings per episode (different noise)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start_noise", type=float, default=0.03, help="dart: std of start-pose noise per arm joint (rad)")
ap.add_argument("--noise_std", type=float, default=0.02, help="dart: stationary std of the OU noise (rad)")
ap.add_argument("--noise_theta", type=float, default=0.05, help="dart: OU mean reversion per frame")
ap.add_argument("--jpeg_quality", type=int, default=95)
ap.add_argument("--gripper_drive", choices=["recorded", "usd"], default="recorded")
ap.add_argument("--no_tray_collider", dest="tray_collider", action="store_false")
ap.add_argument("--render_updates", type=int, default=2,
                help="renders before reading the cameras each frame (the annotators can return the previous frame)")
ap.add_argument("--no_tilt", dest="tilt", action="store_false",
                help="don't apply the episode's rect_tilt_deg to the rect light (eval_isaac.py doesn't)")
ap.add_argument("--headless", default="true")
args = ap.parse_args()

try:
    from isaacsim import SimulationApp
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": args.headless.lower() == "true", "width": 1280, "height": 720})

import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

try:  # Isaac Sim 5.x / 6.x
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation as SingleArticulation
    from omni.isaac.core.prims import RigidPrim as SingleRigidPrim
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.sensor import Camera

# --- scene constants and setup: as in eval_isaac.py ---------------------------------------------
ROBOT = "/World/FR3Duo_withV_Mkit__1_"
BOX = "/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01"
TRAY = "/World/ServingTray"
CAMS = {
    "observation.images.external_camera": f"{ROBOT}/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam",
    "observation.images.left_wrist_camera": f"{ROBOT}/left_d405_color_frame/left_wrist_cam",
    "observation.images.right_wrist_camera": f"{ROBOT}/right_d405_color_frame/right_wrist_cam",
}
CAM_PARAMS = {"observation.images.external_camera": "head", "observation.images.left_wrist_camera": "left_wrist",
              "observation.images.right_wrist_camera": "right_wrist"}
PHYSICS_HZ = 60
ARM = [*range(7), *range(8, 15)]
REF = json.loads(Path(args.ref).read_text())
GC = REF["gripper_contract"]
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def gripper_action_to_joint(a):
    return GC["closed_joint_rad"] * (1.0 - float(np.clip(a, 0, 1)))


def gripper_joint_to_state(q):
    return float(np.interp(q, [0.0, GC["hold_joint_rad"], GC["closed_joint_rad"]], [1.0, GC["hold_norm"], 0.0]))


omni.usd.get_context().open_stage(args.usd)
for _ in range(20):
    app.update()
stage = omni.usd.get_context().get_stage()


def add_tray_collider(interior=(0.229, 0.307), floor_thickness=0.02, wall_height=0.03, wall_thickness=0.01):
    r = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(TRAY)).ComputeAlignedRange()
    mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
    c = (mn + mx) / 2
    top = mx[1]
    ix, iz = interior
    parts = {
        "floor": ((c[0], top - floor_thickness / 2, c[2]), (ix + 2 * wall_thickness, floor_thickness, iz + 2 * wall_thickness)),
        "wall_x0": ((c[0] - ix / 2 - wall_thickness / 2, top + wall_height / 2, c[2]), (wall_thickness, wall_height, iz)),
        "wall_x1": ((c[0] + ix / 2 + wall_thickness / 2, top + wall_height / 2, c[2]), (wall_thickness, wall_height, iz)),
        "wall_z0": ((c[0], top + wall_height / 2, c[2] - iz / 2 - wall_thickness / 2), (ix, wall_height, wall_thickness)),
        "wall_z1": ((c[0], top + wall_height / 2, c[2] + iz / 2 + wall_thickness / 2), (ix, wall_height, wall_thickness)),
    }
    for name, (ctr, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"/World/TrayCollider/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, ctr)))
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def prepare_stage():
    box = stage.GetPrimAtPath(BOX)
    UsdPhysics.MassAPI.Apply(box).CreateMassAttr(0.10)
    mat = UsdShade.Material.Define(stage, Sdf.Path("/World/PhysicsMaterials/box_material"))
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(0.9)
    pm.CreateDynamicFrictionAttr(0.8)
    pm.CreateRestitutionAttr(0.0)
    for p in [box, *box.GetAllChildren()]:
        if p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    if args.gripper_drive == "recorded":
        d = REF["gripper_drive_recorded"]
        for side in ["left", "right"]:
            j = stage.GetPrimAtPath(f"{ROBOT}/{side}_robotiq_85_base_link/{side}_gripper_ctrl")
            drv = UsdPhysics.DriveAPI.Get(j, "angular")
            drv.GetStiffnessAttr().Set(d["stiffness"] * np.pi / 180)
            drv.GetDampingAttr().Set(d["damping"] * np.pi / 180)
            drv.GetMaxForceAttr().Set(d["maxForce"])
    if args.tray_collider:
        add_tray_collider()
    for key, path in CAMS.items():
        cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        p = REF["camera_params"][CAM_PARAMS[key]]
        cam.GetFocalLengthAttr().Set(p["focal_length"])
        cam.GetHorizontalApertureAttr().Set(p["horizontal_aperture"])
        cam.GetVerticalApertureAttr().Set(p["vertical_aperture"])


TILT_OP = None


def set_lighting(cfg):
    """Intensity and colour temperature as eval_isaac.py, plus the episode's rect light tilt (rect_tilt_deg, x/y
    degrees on top of the light's own orientation) — check with --mode snapshot + `hackpipe sim camera-gap`."""
    global TILT_OP
    rect, dome = UsdLux.RectLight(stage.GetPrimAtPath("/World/RectLight")), UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    for light, c in [(rect, cfg["rect"]), (dome, cfg["dome"])]:
        light.GetIntensityAttr().Set(float(c["intensity_to"]))
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(c["color_temperature_K"]))
    tilt = cfg.get("rect_tilt_deg")
    if args.tilt and tilt is not None:
        if TILT_OP is None:
            TILT_OP = UsdGeom.Xformable(rect.GetPrim()).AddRotateXYOp(opSuffix="hackpipeTilt")
        TILT_OP.Set(Gf.Vec2f(float(tilt[0]), float(tilt[1])))


def render():
    for _ in range(max(1, args.render_updates)):
        world.render()


prepare_stage()
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ)
arms = {s: world.scene.add(SingleArticulation(prim_path=f"{ROBOT}/{s}_arm_root_joint", name=f"{s}_arm")) for s in ["left", "right"]}
box = world.scene.add(SingleRigidPrim(prim_path=BOX, name="box"))
cams = {key: Camera(prim_path=path, resolution=(224, 224), frequency=PHYSICS_HZ) for key, path in CAMS.items()}
world.reset()
for c in cams.values():
    c.initialize()
idx = {s: [arms[s].get_dof_index(f"{s}_fr3v2_joint{i}") for i in range(1, 8)] + [arms[s].get_dof_index(f"{s}_gripper_ctrl")]
       for s in ["left", "right"]}
tray_box = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetPrimAtPath(TRAY)).ComputeAlignedRange()
TRAY_MIN, TRAY_MAX = np.array(tray_box.GetMin()), np.array(tray_box.GetMax())


def read_state():
    out = []
    for s in ["left", "right"]:
        q = arms[s].get_joint_positions()[idx[s]]
        out += list(q[:7]) + [gripper_joint_to_state(q[7])]
    return np.array(out, np.float32)


def apply_target(a16):
    for k, s in enumerate(["left", "right"]):
        a = a16[8 * k: 8 * k + 8]
        arms[s].apply_action(ArticulationAction(joint_positions=np.r_[a[:7], gripper_action_to_joint(a[7])],
                                                joint_indices=np.array(idx[s])))


def reset_robot(state16):
    for k, s in enumerate(["left", "right"]):
        q = arms[s].get_joint_positions().copy()
        q[idx[s][:7]] = state16[8 * k: 8 * k + 7]
        q[idx[s][7]] = 0.0
        arms[s].set_joint_positions(q)
        arms[s].set_joint_velocities(np.zeros_like(q))
    apply_target(np.array(state16, np.float32))


def images():
    out = {}
    for key, c in cams.items():
        rgba = c.get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError(f"camera {key} returned no image (render the stage first)")
        rgb = rgba[..., :3]
        out[key] = (rgb * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    return out


def box_pose():
    p, q = box.get_world_pose()
    return np.asarray(p, float), np.asarray(q, float)


def box_tilt_deg(q):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return float(np.degrees(np.arccos(np.clip(np.abs(R.T @ np.array([0, 1, 0])).max(), -1, 1))))


def on_tray(p, margin=0.01):
    return bool(np.all(p[[0, 2]] > TRAY_MIN[[0, 2]] + margin) and np.all(p[[0, 2]] < TRAY_MAX[[0, 2]] - margin)
                and TRAY_MIN[1] < p[1] < TRAY_MAX[1] + 0.15)


def settle(n=30):
    for _ in range(n):
        world.step(render=False)


def begin(ep, init):
    world.reset()
    set_lighting(ep["light"])
    reset_robot(init)
    settle()


# --- modes ----------------------------------------------------------------------------------------
splits = set(args.split.split(","))
episodes = [e for e in REF["record_episodes"] if e["split"] in splits][: args.limit]
A = np.array(REF["demo_actions"], np.float32)
init0 = np.array(REF["initial_state"], np.float32)
print(f"{args.mode}: {len(episodes)} episodes x {args.repeats if args.mode == 'dart' else 1}, {len(A)} frames", flush=True)

for ep in episodes:
    if args.mode == "snapshot":
        begin(ep, init0)
        render()
        d = OUT / "snapshots" / str(ep["episode_index"])
        d.mkdir(parents=True, exist_ok=True)
        for key, img in images().items():
            Image.fromarray(img).save(d / f"{key.split('.')[-1]}.png")
        (d / "state.json").write_text(json.dumps({"sim_state": read_state().round(5).tolist(),
                                                  "dataset_state": REF["initial_state"]}))
        print("snapshot", ep["episode_index"], flush=True)
        continue
    for rep in range(args.repeats if args.mode == "dart" else 1):
        rng = np.random.default_rng([args.seed, int(ep["episode_index"]), rep])
        init = init0.copy()
        if args.mode == "dart":
            init[ARM] += rng.normal(0, args.start_noise, len(ARM)).astype(np.float32)
        begin(ep, init)
        name = f"{ep['name'] or ep['episode_index']}" + (f"_r{rep}" if args.mode == "dart" else "")
        d = OUT / f"{args.mode}_{int(ep['episode_index']):03d}_{rep}"
        cam_dirs = {key: d / "cams" / key.split(".")[-1] for key in CAMS}
        for p in cam_dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        p0, _ = box_pose()
        S, ou, t0 = [], np.zeros(len(ARM), np.float32), time.time()
        sigma = args.noise_std * np.sqrt(2 * args.noise_theta)          # stationary std = noise_std
        max_lift = 0.0
        for k in range(len(A)):
            render()
            for key, img in images().items():
                Image.fromarray(img).save(cam_dirs[key] / f"{k:06d}.jpg", quality=args.jpeg_quality)
            S.append(read_state())
            target = A[k].copy()                 # recorded label: the demo's target (also under noise)
            if args.mode == "dart":
                ou += -args.noise_theta * ou + sigma * rng.normal(0, 1, len(ARM)).astype(np.float32)
                target[ARM] += ou
            apply_target(target)
            world.step(render=False)
            world.step(render=False)
            max_lift = max(max_lift, box_pose()[0][1] - p0[1])
        settle(60)
        p, q = box_pose()
        ok = on_tray(p)
        np.save(d / "state.npy", np.array(S, np.float32))
        np.save(d / "action.npy", A)
        meta = {"name": name, "mode": args.mode, "source": ep["source"], "source_episode": ep["episode_index"],
                "split_of_source": ep["split"], "seed": ep["seed"], "light": ep["light"], "success": ok,
                "stage": 4 if ok else (1 if max_lift > 0.03 else 0), "final_box_pose": [*p.round(5).tolist(), *q.round(5).tolist()],
                "box_tilt_deg": round(box_tilt_deg(q), 2), "max_lift_mm": round(max_lift * 1000, 1),
                "light_tilt_applied": bool(args.tilt and ep["light"].get("rect_tilt_deg") is not None),
                "noise": {"start": args.start_noise, "std": args.noise_std, "theta": args.noise_theta, "rep": rep,
                          "seed": args.seed} if args.mode == "dart" else None,
                "recorder": "scripts/isaac/record_demos.py", "usd": Path(args.usd).name,
                "reference": REF.get("hackpipe"), "wall_s": round(time.time() - t0, 1)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print("EPISODE", json.dumps({k: meta[k] for k in ("name", "success", "stage", "max_lift_mm", "wall_s")}), flush=True)

app.close()
