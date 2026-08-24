"""16 — Detector showcase: four detectors, each in its own regime, live throughput.

Renders an animated clip of four detector technologies, each simulated in the
observing regime it is actually used in — its own scene, band, telescope,
exposure and readout mode:

  1. CCD    (``andor_ikon_m934``)                 -- deep-cooled scientific CCD
                                                     on a 5 s deep-sky field:
                                                     effectively zero dark, so
                                                     the faint end is set by the
                                                     output-amplifier read noise
  2. EMCCD  (``andor_ocam2k``)                     -- stochastic electron
                                                     multiplication at x600
                                                     running as a 500 Hz AO
                                                     wavefront sensor: single
                                                     photons become visible, at
                                                     the cost of excess-noise
                                                     speckle
  3. sCMOS  (``andor_marana_4_2b_11``)             -- 100 ms wide-field imaging,
                                                     where the per-pixel read
                                                     noise gives the noise floor
                                                     itself a fixed structure
                                                     that repeats frame to frame
  4. eAPD   (``first_light_imaging_cred_one``)     -- C-RED One HgCdTe avalanche
                                                     array doing near-infrared
                                                     AO in H band, read in
                                                     correlated double sampling
                                                     at its 1750 Hz maximum
                                                     frame rate (1/3500 s of
                                                     integration: the other half
                                                     of the frame period is the
                                                     reset and pedestal read)

An earlier version of this clip put all four detectors on one shared V-band
field at a shared 200 ms exposure. That is a tidier controlled comparison, but
it is not a fair showcase: a near-infrared avalanche array carrying a realistic
dark-plus-background ceiling is dark-noise dominated in a 200 ms visible
exposure, and no exposure time fixes it, because amplified dark noise grows as
``sqrt(t)`` while the star signal grows as ``t``. Each panel therefore gets the
scene its detector is built for, and the panel captions state what changed.

A seeded random walk in ``offset_xy`` adds telescope pointing jitter, so each
field visibly moves while the noise re-draws.

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
from dataclasses import dataclass

import numpy as np

import getframes as gf

SHAPE = (256, 256)
SEED = 3
N_STARS = 35
JITTER_RMS_PX = 1.2  # telescope pointing jitter (random walk)


@dataclass(frozen=True)
class PanelSpec:
    """One detector, and the observing regime it is shown in."""

    label: str
    preset: str
    regime: str  # caption line: what this panel is doing
    band: str  # Vega-system Johnson band
    aperture_m: float
    exposure_s: float
    mag_range: tuple[float, float]
    sky_mag: float  # mag/arcsec^2 in `band`
    readout: str  # "expose" or "cds"


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(
        label="CCD",
        preset="andor_ikon_m934",
        regime="deep-sky · 5 s · V · 2.5 m",
        band="V",
        aperture_m=2.5,
        exposure_s=5.0,
        mag_range=(18.0, 21.5),
        sky_mag=21.5,
        readout="expose",
    ),
    PanelSpec(
        label="EMCCD",
        preset="andor_ocam2k",
        regime="AO wavefront sensor · 500 Hz · V · 2.5 m",
        band="V",
        aperture_m=2.5,
        exposure_s=0.002,
        mag_range=(11.0, 15.0),
        sky_mag=20.5,
        readout="expose",
    ),
    PanelSpec(
        label="sCMOS",
        preset="andor_marana_4_2b_11",
        regime="wide-field · 100 ms · V · 2.5 m",
        band="V",
        aperture_m=2.5,
        exposure_s=0.1,
        mag_range=(15.0, 19.0),
        sky_mag=20.5,
        readout="expose",
    ),
    PanelSpec(
        label="eAPD",
        preset="first_light_imaging_cred_one",
        # 1/3500 s of integration is a 1750 Hz CDS frame: the other half of the
        # frame period is the reset and the pedestal read.
        regime="near-IR AO · CDS 1750 Hz · H · 8 m",
        band="H",
        aperture_m=8.0,
        exposure_s=1.0 / 3500.0,
        mag_range=(8.0, 12.0),
        sky_mag=13.5,  # H-band sky is far brighter than V
        readout="cds",
    ),
)

# --- the movie ---------------------------------------------------------------
N_FRAMES = 120
PLAYBACK_FPS = 25
BENCH_WARMUP = 10  # untimed frames before the throughput measurement
BENCH_SECONDS = 1.5  # timed window per panel
DISPLAY_PERCENTILE = 99.95


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


def build_scene(spec: PanelSpec) -> gf.Scene:
    """The star field this detector is shown observing."""
    rng = np.random.default_rng(SEED)
    sources = [
        gf.PointSource(
            x=float(rng.uniform(12, SHAPE[1] - 12)),
            y=float(rng.uniform(12, SHAPE[0] - 12)),
            magnitude=float(rng.uniform(*spec.mag_range)),
        )
        for _ in range(N_STARS)
    ]
    return gf.Scene(
        shape=SHAPE,
        optics=gf.Telescope(
            aperture_diameter_m=spec.aperture_m,
            throughput=0.3,
            plate_scale_arcsec_per_pixel=0.4,
            band=gf.Bandpass.johnson(spec.band),
        ),
        psf=gf.MoffatPSF(fwhm_arcsec=1.4, beta=3.0),
        sources=sources,
        sky=gf.Sky(surface_brightness_mag_arcsec2=spec.sky_mag),
    )


def jitter_track(n_frames: int) -> np.ndarray:
    """A seeded random walk of ``(dx, dy)`` pointing offsets in pixels."""
    rng = np.random.default_rng(SEED + 1)
    steps = rng.normal(0.0, JITTER_RMS_PX / np.sqrt(n_frames), size=(n_frames, 2))
    track = np.cumsum(steps, axis=0)
    return track - track.mean(axis=0)


class Panel:
    """One detector technology observing the field it is built for."""

    def __init__(self, spec: PanelSpec, device: str):
        self.spec = spec
        self.label = spec.label
        self.preset = spec.preset
        self.device = device
        self.fps: float | None = None

        config = gf.load_preset(spec.preset)
        self.sensor = config.sensor_type.value
        self.config = config.replace(resolution=SHAPE)
        self.scene = build_scene(spec)
        self.pedestal_adu = self.measure_pedestal()

    def build_camera(self) -> gf.Camera:
        return gf.Camera(self.config, seed=SEED, precision="float32", device=self.device)

    def read(self, camera: gf.Camera, rate, seed: int | None = None) -> gf.Frame:
        """One frame in this panel's readout mode."""
        if self.spec.readout == "cds":
            return camera.correlated_double_sample(rate, self.spec.exposure_s, seed=seed)
        return camera.expose(rate, self.spec.exposure_s, seed=seed)

    def measure_pedestal(self) -> float:
        """The flat offset a dark subtraction removes, in ADU.

        Bias alone is not the zero point once a detector carries real dark
        current through a gain stage, and a CDS frame sits on its own
        exposure-dependent bias-rate pedestal rather than on the bias offset at
        all. Measuring it as the median of a dark frame taken in this panel's
        own readout mode keeps the physics in the library rather than restating
        it here, and subtracting only the *median* removes the flat pedestal
        while leaving per-pixel dark structure and hot pixels visible, which is
        detector behaviour the clip is meant to show.
        """
        camera = self.build_camera()
        if self.spec.readout == "cds":
            dark = camera.correlated_double_sample(0.0, self.spec.exposure_s, seed=SEED)
        else:
            dark = camera.dark_frame(self.spec.exposure_s, seed=SEED)
        return float(np.median(gf.to_numpy(dark.data)))


def render_rate_maps(panel: Panel, n_frames: int) -> list:
    """Per-frame incident photon-rate maps for this panel, on the compute device.

    Rendered once, outside every timed region: the clip is about detector
    throughput, not about how fast a scene rasterises.
    """
    backend = gf.get_backend(panel.device)
    sky = panel.scene.sky_photon_rate()
    maps = []
    for dx, dy in jitter_track(n_frames):
        rate = panel.scene.photon_rate_map(offset_xy=(float(dx), float(dy)), dtype=np.float32)
        maps.append(backend.asarray(rate + sky, dtype=backend.xp.float32))
    return maps


def benchmark_panel(panel: Panel, rate, sync: Callable[[], None]) -> float:
    """Warm frames/s for this camera, on a fixed device-resident rate map."""
    camera = panel.build_camera()
    for _ in range(BENCH_WARMUP):
        panel.read(camera, rate)
    sync()
    frames = 0
    start = time.perf_counter()
    while True:
        panel.read(camera, rate)
        frames += 1
        if time.perf_counter() - start >= BENCH_SECONDS:
            break
    sync()
    return frames / (time.perf_counter() - start)


def collect_frames(panel: Panel, rates) -> np.ndarray:
    """``(n_frames, *SHAPE)`` of dark-subtracted signal, on the host."""
    camera = panel.build_camera()
    out = np.empty((len(rates), *SHAPE), dtype=np.float32)
    for index, rate in enumerate(rates):
        frame = panel.read(camera, rate, seed=index)
        out[index] = gf.to_numpy(frame.data).astype(np.float32)
    return np.clip(out - panel.pedestal_adu, 0.0, None)


def display_vmax(stack: np.ndarray, config: gf.CameraConfig) -> float:
    """Upper display limit for one panel, set by the scene and not by defects.

    A detector's hot pixels sit at ``hot_pixel_factor`` times its dark rate, and
    behind an avalanche gain stage that puts them orders of magnitude above any
    star in the field. A fixed high percentile is not safe against that, because
    a preset's declared ``hot_pixel_fraction`` can *be* the percentile —
    ``leonardo_saphira``'s 0.05% is exactly the 99.95th — at which point the
    panel is scaled to its own hot pixels and everything physical crushes to
    black. Take the percentile from below the declared defect population
    instead, keeping a 4x margin for its spread.
    """
    defect_fraction = float(config.hot_pixel_fraction or 0.0)
    upper = min(DISPLAY_PERCENTILE, 100.0 - 400.0 * defect_fraction)
    return float(np.percentile(stack, upper)) or 1.0


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
    fig.subplots_adjust(left=0.015, right=0.985, top=0.795, bottom=0.115, wspace=0.03, hspace=0.26)

    images = []
    for ax, panel, stack in zip(axes.flat, panels, frames):
        ax.set_facecolor(bg)
        # Per-panel scale on dark-subtracted signal: the four detectors differ in
        # QE, gain and bias pedestal, and now in scene and exposure too, so one
        # shared ADU range would say almost nothing about their noise.
        vmax = display_vmax(stack, panel.config)
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
        ax.set_title(panel.label, color=ink, fontsize=13, fontweight="bold", pad=22)
        ax.text(
            0.5,
            1.012,
            panel.spec.regime,
            transform=ax.transAxes,
            color=sub,
            fontsize=8.5,
            ha="center",
            va="bottom",
        )
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
        f"four detector technologies · {SHAPE[0]}² · each in the regime it is "
        f"built for · {device_label}",
        color=sub,
        fontsize=10.5,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.895,
        "own scene, band, telescope, exposure and readout (photon → electron → ADU)",
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
    cb.set_label(
        f"dark-subtracted signal [ADU] — {panels[0].label} scale (each panel scaled independently)",
        color=sub,
        fontsize=8,
    )
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

    panels = [Panel(spec, device) for spec in PANELS]

    print(f"rendering {args.frames} jittered photon-rate maps x {len(panels)} panels ...")
    rates = [render_rate_maps(panel, args.frames) for panel in panels]

    print("benchmarking throughput per panel ...")
    for panel, panel_rates in zip(panels, rates):
        panel.fps = benchmark_panel(panel, panel_rates[0], sync)
        print(f"  {panel.label:6s} {panel.preset:30s} {panel.fps:9,.0f} frames/s")

    print(f"exposing {args.frames} frames x {len(panels)} panels ...")
    frames = [collect_frames(panel, panel_rates) for panel, panel_rates in zip(panels, rates)]

    out = render_animation(panels, frames, label, args.out, PLAYBACK_FPS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
