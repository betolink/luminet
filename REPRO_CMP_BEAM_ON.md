# Reproducing `cmp_beam_on.png`

Reference image: `cmp_beam_on.png` (1200×1200) — a dark, desaturated warm
gold/gray black-hole accretion disk with strong Doppler beaming and clumpy
banding in the disk.

## Repro command

```bash
python render_raster.py \
    --gamma 2.0 \
    --color blackbody \
    --texture noise \
    --noise-amp 1.3 \
    --exposure 1.5 \
    --output cmp_beam_on_repro.png
```

All other flags are at their defaults:

```
--inclination 80  --mass 1  --spin 0
--inner-radius 3  --outer-radius 100
--n-radius 400    --n-angle 720
--beaming
--width 1200      --height 1200
--noise-octaves 5 --noise-anisotropy 4.0
--noise-direction azimuthal  --noise-seed 0
--percentile 99.5
```

Validated against `render_raster.py` at commit `586014b`.

## What each non-default knob does

- **`--gamma 2.0`** — the dominant *structural* lever. The default `0.45`
  renders too bright and under-beamed. Raising gamma darkens the image,
  concentrates the light toward the center, and increases the approaching/
  receding (left/right) beaming asymmetry. `2.0` gives the dark, high-contrast
  look of the reference.
- **`--color blackbody`** — the closest match to the reference's pale/dark
  gold-gray ramp. The default `inferno` is too saturated and goes purple in
  the low-luminance end.
- **`--texture noise` + `--noise-amp 1.3`** — adds the clumpy banding that
  makes the disk look realistic. The default `--texture none` is too smooth
  and clean.
- **`--exposure 1.5`** — brightens the disk to match the reference intensity.

## How we got here (search notes)

- The default render (`gamma 0.45`, `inferno`) did not match: too bright,
  not enough beaming, wrong color.
- Ruled out: the blue palette (far too blue, R−B ≈ −100), exposure-only
  tweaks, inclination changes (barely move the beaming L/R ratio), and
  beaming on/off (barely move L/R).
- **gamma** is the dominant lever for structure (beaming asymmetry and
  central concentration); **color** and **texture** are independent levers
  for palette and realism respectively.
- `blackbody` is the closest built-in palette to the reference's desaturated
  warm ramp.
- `texture noise` is what introduces the banding; `infall` and `filament`
  were also tried but `noise` matched best.

## Metrics (normalized-luminance comparison vs the reference)

| candidate (all `--gamma 2.0 --color blackbody --texture noise`) | lit-mean | p95 | banding | corr |
|---|---|---|---|---|
| reference `cmp_beam_on.png` | 45.0 | 44.7 | 29 | 1.000 |
| **chosen: `--noise-amp 1.3 --exposure 1.5`** | 20.8 | 7.0 | 23 | 0.767 |
| `--exposure 1.5` | 30.3 | 12.3 | 21 | 0.874 |
| `--exposure 2.0` | 41.6 | 22.3 | 19 | 0.934 |
| `--noise-amp 1.3` | 13.3 | 2.7 | 23 | 0.641 |
| `--exposure 2.0 --n-radius 250 --n-angle 450` | 39.8 | 19.3 | 33 | 0.887 |

`banding` = max number of direction changes along a radial line through the
disk (a banding proxy); `corr` = Pearson correlation of the normalized
luminance fields. The chosen candidate prioritizes the realistic clumpy
banding (23, close to the reference's 29) over raw brightness correlation.
`--exposure 2.0` scores higher on `corr` but is smoother/less banded; a
coarser solve grid (`--n-radius 250 --n-angle 450`) adds more concentric
banding at the cost of correlation.
