# Record frames via Replicator and print an ffmpeg stitch command
# with auto-stamped output dirs and clean shutdown.

import os, glob, re, shutil, time, errno, stat
from datetime import datetime
from typing import Tuple, Optional

_STATE = {"writer": None, "render_product": None, "out_dir": None, "res": (1280, 720)}

def _nowstamp():
    # YYYY-mm-dd_HH-MM-SS (safe for file names)
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _ensure_camera() -> str:
    import omni.replicator.core as rep
    from omni.usd import get_context
    from pxr import UsdGeom, Gf

    stage = get_context().get_stage()
    if stage.GetPrimAtPath("/OmniverseKit_Persp"):
        return "/OmniverseKit_Persp"
    cam_path = "/World/RecCam"
    UsdGeom.Camera.Define(stage, cam_path)
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cam_path))
    xf.SetTranslate(Gf.Vec3d(2.5, -2.5, 2.0))
    xf.SetRotate(Gf.Vec3f(0.0, 0.0, 35.0))
    return cam_path

def _mk_run_dir(base_dir: str, prefix: str = "forward_test_frames") -> str:
    """
    Create a unique run-stamped directory under base_dir and a 'latest' symlink.
    """
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, f"{prefix}_{_nowstamp()}")
    os.makedirs(run_dir, exist_ok=False)
    # Update 'latest' symlink for convenience
    latest = os.path.join(base_dir, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.unlink(latest)
        os.symlink(run_dir, latest)
    except OSError:
        # Not fatal (e.g., on Windows or if permissions block symlinks)
        pass
    return run_dir

def start_frames(
    out_dir: Optional[str] = None,
    resolution: Tuple[int, int] = (1280, 720),
    prefix: str = "forward_test_frames",
):
    """
    Begin frame recording.
    If out_dir is None, uses $REPLICATOR_REC_DIR or '/workspace/seamlis/recordings',
    and creates a stamped subfolder like 'forward_test_frames_YYYY-mm-dd_HH-MM-SS'.
    Also writes/updates a 'latest' symlink in the base directory.
    """
    import omni.replicator.core as rep

    if out_dir is None:
        base_dir = os.environ.get("REPLICATOR_REC_DIR", "/workspace/seamlis/recordings")
        out_dir = _mk_run_dir(base_dir, prefix=prefix)
    else:
        # If a full path is provided, ensure it exists but don't stamp
        os.makedirs(out_dir, exist_ok=True)

    cam = _ensure_camera()
    rp = rep.create.render_product(cam, tuple(resolution))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=out_dir, rgb=True)
    writer.attach([rp])
    _STATE.update({"writer": writer, "render_product": rp, "out_dir": out_dir, "res": resolution})
    print(f"[rec] Writing frames --> {out_dir}  ({resolution[0]}x{resolution[1]})")

def _detect_start_number(out_dir: str, prefix="rgb_", ext=".png", pad=4) -> int:
    pattern = os.path.join(out_dir, f"{prefix}{'?'*pad}{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        return 0
    m = re.search(rf"{re.escape(prefix)}(\d{{{pad}}}){re.escape(ext)}$", os.path.basename(files[0]))
    return int(m.group(1)) if m else 0

def print_ffmpeg_cmd_for_dt(out_dir: str, sim_dt: float, prefix="rgb_", pad=4, ext=".png",
                            out_mp4: Optional[str]=None):
    fps = 30 if sim_dt <= 0 else round(1.0 / sim_dt)
    start_num = _detect_start_number(out_dir, prefix=prefix, ext=ext, pad=pad)
    if out_mp4 is None:
        out_mp4 = f"{out_dir}.mp4"
    seq = os.path.join(out_dir, f"{prefix}%0{pad}d{ext}")
    cmd = (
        "ffmpeg -y "
        f"-framerate {fps} "
        f"-start_number {start_num} "
        f"-i {seq} "
        "-c:v libx264 -pix_fmt yuv420p "
        f"{out_mp4}"
    )
    print("\n[rec] To stitch frames -> MP4 at real-time speed:")
    print(cmd)
    print("[rec] Tip: adjust -framerate if intentionally want faster/slower playback.\n")

def _onerror_make_writable(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        func(path)
    except Exception:
        pass

def _safe_delete_folder(path: str):
    try:
        shutil.rmtree(path, onerror=_onerror_make_writable)
        return True
    except PermissionError:
        user = os.environ.get("USER", "my-username")
        print(
            f"[rec] Could not delete due to permissions. If files were created as root,\n"
            f"      run and retry:\n"
            f"      sudo chown -R {user}:{user} {path}\n"
            f"      rm -rf {path}\n"
        )
        return False
    except FileNotFoundError:
        return True
    except Exception as e:
        print(f"[rec] Delete failed: {e}")
        return False

def stop_frames(sim=None, sim_dt: Optional[float]=None, *, delete: bool=False, wait_sec: float=0.25):
    """
    Clean stop: detach writer, flush, print ffmpeg, and optionally delete the frame folder.
    - delete=True will attempt to remove the output directory after printing the ffmpeg command.
    """
    out_dir = _STATE.get("out_dir")
    writer = _STATE.get("writer")
    rp = _STATE.get("render_product")

    if not out_dir:
        print("[rec] No active recording.")
        return

    # Determine dt
    if sim_dt is None and sim is not None and hasattr(sim, "get_physics_dt"):
        try:
            sim_dt = float(sim.get_physics_dt())
        except Exception:
            sim_dt = None
    if sim_dt is None:
        sim_dt = 1.0 / 30.0

    # Detach & shutdown writer
    try:
        if writer is not None and rp is not None:
            try:
                writer.detach([rp])
            except Exception:
                pass
            if hasattr(writer, "shutdown"):
                try:
                    writer.shutdown()
                except Exception:
                    pass
    finally:
        time.sleep(max(0.0, wait_sec))  # let FS flush

    print(f"[rec] Stopped. Frames are in: {out_dir}")
    print_ffmpeg_cmd_for_dt(out_dir, sim_dt)

    if delete:
        ok = _safe_delete_folder(out_dir)
        if ok:
            print(f"[rec] Deleted frame folder: {out_dir}")
        else:
            print(f"[rec] Frame folder not removed. See guidance above.")

    _STATE.clear()

class frame_recording:
    """Context manager:
        with frame_recording(base_dir="/workspace/seamlis/recordings", delete=True):
            # run sim ...
    Creates stamped subfolder automatically when base_dir is given.
    """
    def __init__(self, out_dir: Optional[str]=None, resolution=(1280,720), delete=False, prefix="forward_test_frames"):
        self.out_dir = out_dir
        self.resolution = resolution
        self.delete = delete
        self.prefix = prefix

    def __enter__(self):
        if self.out_dir is None:
            base_dir = os.environ.get("REPLICATOR_REC_DIR", "/workspace/seamlis/recordings")
            stamped = _mk_run_dir(base_dir, prefix=self.prefix)
            start_frames(stamped, self.resolution)
        else:
            start_frames(self.out_dir, self.resolution)
        return self

    def __exit__(self, exc_type, exc, tb):
        stop_frames(delete=self.delete)
        return False

def prune_old_runs(base_dir: str, keep_last_n: int = 3, prefix: str = "forward_test_frames"):
    """
    Keep only the newest N run folders under base_dir matching 'prefix_*'.
    """
    if not os.path.isdir(base_dir):
        return
    entries = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix + "_")]
    entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for obsolete in entries[keep_last_n:]:
        _safe_delete_folder(obsolete)
