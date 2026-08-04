"""16 — Detector showcase: one photon field, four detectors, live throughput.

Renders an animated clip of the *same* incident photon field landing on four
detector technologies, so the difference in the frames is the detector physics
and nothing else:

  1. CCD    (``andor_ikon_m934``)        -- deep-cooled scientific CCD; a clean
                                            frame whose faint end is set by the
                                            output-amplifier read noise
  2. EMCCD  (``andor_ocam2k``)           -- stochastic electron multiplication
                                            at x600: single photons become
                                            visible, at the cost of
                                            excess-noise speckle
  3. sCMOS  (``andor_marana_4_2b_11``)   -- per-pixel read noise, so the noise
                                            floor itself has fixed structure
                                            that repeats frame to frame
  4. eAPD   (``leonardo_saphira``)       -- HgCdTe avalanche array, the fast
                                            low-noise AO wavefront-sensing path

``Scene.photon_rate_map`` produces the incident rate *before* quantum
efficiency, so all four cameras see one physical photon field and each applies
its own QE, gain stage, and noise. A seeded random walk in ``offset_xy`` adds
telescope pointing jitter, so the field visibly moves while the noise re-draws.

Each panel is overlaid with the frame rate that camera sustained on this machine
(warm persistent camera, device-resident rate and output, CUDA synchronized),
matching the methodology of ``benchmarks/bench_devices.py``.

Run:  ``python examples/16_detector_showcase.py``       (auto GPU if available)
      ``python examples/16_detector_showcase.py --device cpu --frames 40``
      ``python examples/16_detector_showcase.py --out docs/assets/showcase.webp``

Needs matplotlib + pillow; a CUDA GPU (``pip install 'getframes[gpu]'``) is what
turns the overlays into the headline.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

import getframes as gf

# --- the scene: one faint star field, shared by every panel ------------------
SHAPE = (256, 256)
# A short exposure is the point: it keeps the EMCCD's x600 gain stage below its
# full well while leaving the unamplified CCD read-noise limited, which is
# exactly the regime where the four technologies stop looking alike.
EXPOSURE_S = 0.2  # s
SEED = 3
N_STARS = 35
MAG_RANGE = (15.5, 19.0)  # bright enough to see structure, faint enough to see noise
SKY_MAG = 20.5  # mag/arcsec^2
JITTER_RMS_PX = 1.2  # telescope pointing jitter (random walk)

PANELS = (
    ("CCD", "andor_ikon_m934"),
    ("EMCCD", "andor_ocam2k"),
    ("sCMOS", "andor_marana_4_2b_11"),
    ("eAPD", "leonardo_saphira"),
)

# --- the movie ---------------------------------------------------------------
N_FRAMES = 120
PLAYBACK_FPS = 25
BENCH_WARMUP = 10  # untimed frames before the throughput measurement
BENCH_SECONDS = 1.5  # timed window per panel


def _resolve_device(requested: str) -> str:
    """``"auto"`` -> ``"gpu"`` when CuPy imports, else ``"cpu"``."""
    if requested != "auto":
        return requested
    try:
        import cupy  # noqa: F401

        return "gpu"
    except Exception:
        return "cpu"


def _device_label(device: str) -> str:
    if device == "gpu":
        try:
            import cupy as cp

            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
            return name.replace("NVIDIA GeForce ", "").strip()
        except Exception:
            return "GPU"
    import platform

    return f"CPU ({platform.machine()})"


def _make_sync(device: str) -> Callable[[], None]:
    """A no-op on CPU; a full device barrier on GPU (so timings are honest)."""
    if device != "gpu":
        return lambda: None
    import cupy as cp

    return cp.cuda.Stream.null.synchronize


def build_scene() -> gf.Scene:
    """A faint star field seen through a 2.5 m telescope in V."""
    rng = np.random.default_rng(SEED)
    sources = [
        gf.PointSource(
            x=float(rng.uniform(12, SHAPE[1] - 12)),
            y=float(rng.uniform(12, SHAPE[0] - 12)),
            magnitude=float(rng.uniform(*MAG_RANGE)),
        )
        for _ in range(N_STARS)
    ]
    return gf.Scene(
        shape=SHAPE,
        optics=gf.Telescope(
            aperture_diameter_m=2.5,
            throughput=0.3,
            plate_scale_arcsec_per_pixel=0.4,
            band=gf.Bandpass.johnson("V"),
        ),
        psf=gf.MoffatPSF(fwhm_arcsec=1.4, beta=3.0),
        sources=sources,
        sky=gf.Sky(surface_brightness_mag_arcsec2=SKY_MAG),
    )


def jitter_track(n_frames: int) -> np.ndarray:
    """A seeded random walk of ``(dx, dy)`` pointing offsets in pixels."""
    rng = np.random.default_rng(SEED + 1)
    steps = rng.normal(0.0, JITTER_RMS_PX / np.sqrt(n_frames), size=(n_frames, 2))
    track = np.cumsum(steps, axis=0)
    return track - track.mean(axis=0)


def render_rate_maps(scene: gf.Scene, n_frames: int, device: str) -> list:
    """Per-frame incident photon-rate maps, on the compute device.

    Rendered once, outside every timed region: the clip is about detector
    throughput, not about how fast a scene rasterises.
    """
    backend = gf.get_backend(device)
    sky = scene.sky_photon_rate()
    maps = []
    for dx, dy in jitter_track(n_frames):
        rate = scene.photon_rate_map(offset_xy=(float(dx), float(dy)), dtype=np.float32)
        maps.append(backend.asarray(rate + sky, dtype=backend.xp.float32))
    return maps


class Panel:
    """One detector technology exposed to the shared photon field."""

    def __init__(self, label: str, preset: str, device: str):
        self.label = label
        self.preset = preset
        self.device = device
        self.fps: float | None = None

        config = gf.load_preset(preset)
        self.sensor = config.sensor_type.value
        self.config = config.replace(resolution=SHAPE)
        self.bias_adu = float(self.config.bias_offset_adu)

    def build_camera(self) -> gf.Camera:
        return gf.Camera(self.config, seed=SEED, precision="float32", device=self.device)


def benchmark_panel(panel: Panel, rate, sync: Callable[[], None]) -> float:
    """Warm frames/s for this camera, on a fixed device-resident rate map."""
    camera = panel.build_camera()
    for _ in range(BENCH_WARMUP):
        camera.expose(rate, EXPOSURE_S)
    sync()
    frames = 0
    start = time.perf_counter()
    while True:
        camera.expose(rate, EXPOSURE_S)
        frames += 1
        if time.perf_counter() - start >= BENCH_SECONDS:
            break
    sync()
    return frames / (time.perf_counter() - start)


def collect_frames(panel: Panel, rates) -> np.ndarray:
    """``(n_frames, *SHAPE)`` of signal above bias, on the host."""
    camera = panel.build_camera()
    out = np.empty((len(rates), *SHAPE), dtype=np.float32)
    for index, rate in enumerate(rates):
        frame = camera.expose(rate, EXPOSURE_S, seed=index)
        out[index] = gf.to_numpy(frame.data).astype(np.float32)
    return np.clip(out - panel.bias_adu, 0.0, None)


def render_animation(panels, frames, device_label, out_path, playback_fps):
    """Assemble the four panels into one animated clip via matplotlib + pillow.

    Saved as animated WebP rather than GIF: a quantized GIF palette shared
    across frames bands and flickers on the smooth sky gradient as the photon
    noise drifts across quantization boundaries frame to frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ink, sub, bg = "#e6edf3", "#9aa7b4", "#0b0f14"
    accent = "#ffd166"
    highlight = "#7fd1c1"

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.8), dpi=68)
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.80, bottom=0.115, wspace=0.03, hspace=0.22)

    images = []
    for ax, panel, stack in zip(axes.flat, panels, frames):
        ax.set_facecolor(bg)
        # Per-panel scale on signal above bias: the four detectors differ in QE,
        # gain and bias pedestal, so one shared ADU range would say more about
        # their gain settings than about their noise.
        vmax = float(np.percentile(stack, 99.95)) or 1.0
        im = ax.imshow(
            stack[0],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            origin="lower",
            interpolation="nearest",
            animated=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#233040")
        ax.set_title(panel.label, color=ink, fontsize=13, fontweight="bold", pad=16)
        rate = f"{panel.fps:,.0f} frames/s" if panel.fps else ""
        ax.text(
            0.045,
            0.955,
            rate,
            transform=ax.transAxes,
            color=accent,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.32", "fc": "#11161d", "ec": "#2c3846"},
        )
        ax.text(
            0.955,
            0.045,
            panel.preset,
            transform=ax.transAxes,
            color=highlight,
            fontsize=8.5,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#11161d", "ec": "#2c3846"},
        )
        images.append(im)

    fig.text(
        0.5,
        0.965,
        "getframes — detector showcase",
        color=ink,
        fontsize=17,
        fontweight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.925,
        f"one incident photon field · {SHAPE[0]}² · {EXPOSURE_S * 1e3:.0f} ms · "
        f"V band · sky {SKY_MAG:g} mag/arcsec² · {device_label}",
        color=sub,
        fontsize=10.5,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.895,
        "same photons in, four detector technologies out (photon → electron → ADU)",
        color=sub,
        fontsize=10,
        ha="center",
        va="top",
    )

    tstamp = fig.text(
        0.985, 0.098, "", color=sub, fontsize=10, ha="right", va="bottom", family="monospace"
    )
    fig.text(
        0.015,
        0.098,
        "overlay = live throughput on this machine",
        color=sub,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    cax = fig.add_axes([0.30, 0.048, 0.40, 0.016])
    cb = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    cb.set_label("signal above bias [ADU] (per-panel scale)", color=sub, fontsize=9)
    cb.outline.set_edgecolor("#233040")
    cax.tick_params(colors=sub, labelsize=8)

    def rgba_frame() -> Image.Image:
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")

    n_frames = len(frames[0])
    pil_frames = []
    for index in range(n_frames):
        for im, stack in zip(images, frames):
            im.set_data(stack[index])
        tstamp.set_text(f"frame {index + 1:3d}/{n_frames}")
        pil_frames.append(rgba_frame())
    plt.close(fig)

    pil_frames[0].save(
        out_path,
        format="WEBP",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / playback_fps),
        loop=0,
        quality=90,
        method=6,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--frames",
        type=int,
        default=N_FRAMES,
        help=f"number of animation frames (default {N_FRAMES})",
    )
    parser.add_argument("--out", default="detector_showcase.webp")
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be >= 2")

    device = _resolve_device(args.device)
    label = _device_label(device)
    sync = _make_sync(device)
    print(f"device: {device}  ({label})")

    scene = build_scene()
    print(f"rendering {args.frames} jittered photon-rate maps ...")
    rates = render_rate_maps(scene, args.frames, device)

    panels = [Panel(name, preset, device) for name, preset in PANELS]

    print("benchmarking throughput per panel ...")
    for panel in panels:
        panel.fps = benchmark_panel(panel, rates[0], sync)
        print(f"  {panel.label:6s} {panel.preset:26s} {panel.fps:9,.0f} frames/s")

    print(f"exposing {args.frames} frames x {len(panels)} panels ...")
    frames = [collect_frames(panel, rates) for panel in panels]

    out = render_animation(panels, frames, label, args.out, PLAYBACK_FPS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
