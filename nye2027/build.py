#!/usr/bin/env python3
"""NEW YEAR'S EVE 2027 — teaser build script.

Rebuilds the whole edit from the raw clips:
  python3 build.py --src <folder with clip1.mov … clip7.mov> [--out NYE2027_teaser.mp4]

Pipeline: EDL shots -> graded 1080x1920 segments -> concat -> bloom/vignette/grain
          + PIL-rendered title reveal; soundtrack = clip5 live audio (breakdown -> drop),
          filter sweep, pre-drop breath, hard stop + reverb tail + sub impact.
"""
import argparse, json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.signal import butter, sosfiltfilt, fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
W, H = 1080, 1920
BEAT = 60 / 129.2            # tempo of the soundtrack (clip5), measured with librosa
A0 = 11.90 - 16 * BEAT       # audio in-point: the drop (11.90s in clip5) lands on beat 16
DROP = 16 * BEAT             # 7.43s
END_MUSIC = 30 * BEAT        # 13.94s: music stops on a downbeat, title hits
TOTAL = 19.5
SR = 44100

# Per-source grade: (saturation, gamma, brightness, denoise strength, crop for 4:3 source)
GRADE = {
    1: dict(sat=0.88, gamma=1.06, bri=0.00, dn=3.0, crop="crop=360:640:60:0"),
    2: dict(sat=0.85, gamma=1.02, bri=0.00, dn=2.5, crop=None),
    3: dict(sat=0.80, gamma=0.95, bri=-0.01, dn=1.5, crop=None),
    4: dict(sat=0.74, gamma=0.86, bri=-0.03, dn=1.5, crop=None),
    5: dict(sat=0.70, gamma=0.98, bri=-0.01, dn=1.5, crop=None),
    6: dict(sat=0.82, gamma=0.84, bri=-0.03, dn=1.5, crop=None),
    7: dict(sat=0.78, gamma=0.86, bri=-0.03, dn=1.5, crop=None),
}

# EDL: (start beat, end beat, clip, source in-point seconds, note)
# Cuts land on the beat grid of the soundtrack.
EDL = [
    # INTRO — muffled breakdown, contemplative: arrival, mystery, details
    (0,    3,    2, 0.15,  "dark tunnel, green emblem glowing — fade from black"),
    (3,    6,    6, 0.00,  "red carpet from above, guest with flowers"),
    (6,    8,    7, 0.05,  "gold ring chandelier, balcony of lights"),
    # BUILD — filter opens, cuts tighten
    (8,    10,   3, 2.40,  "gold balloon under the vaulted cellar"),
    (10,   12,   1, 1.00,  "DJ deck detail, hand on the jog"),
    (12,   13,   6, 3.40,  "balcony POV: photographers, crowd"),
    (13,   14,   4, 3.40,  "purple vault packed"),
    (14,   15.15, 3, 15.55, "DJ at the decks, blue light"),
    # 15.15 -> 16 : black (pre-drop breath, matches the gap in the track)
    # CLIMAX — drop
    (16,   18,   4, 0.85,  "DROP: champagne raised, sparklers"),
    (18,   20,   5, 0.00,  "light show over the crowd from the booth"),
    (20,   21,   1, 7.45,  "orange smoke, hands up"),
    (21,   23,   2, 11.70, "strobe: vault crowd / green emblem"),
    (23,   24,   5, 12.00, "cyan/magenta crowd"),
    (24,   25,   3, 16.00, "DJ hands on the mixer"),
    (25,   26,   4, 3.55,  "purple vault"),
    (26,   27,   5, 14.50, "green/red crowd wide"),
    (27,   28,   2, 12.85, "strobe flash, balloons"),
    (28,   30,   3, 21.90, "hero wide: gold balloon, rainbow vault — smash to black"),
]

def fr(t):
    return int(round(t * FPS))

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:]); raise SystemExit(f"failed: {' '.join(cmd[:6])}")

def grade_chain(c):
    g = GRADE[c]
    vf = []
    if g["crop"]:
        vf.append(g["crop"])
    vf += [
        f"hqdn3d={g['dn']}:{g['dn']}:{g['dn']*2}:{g['dn']*2}",
        f"fps={FPS}",
        f"scale={W}:{H}:flags=lanczos",
        "unsharp=5:5:0.45:5:5:0.0",
        f"eq=contrast=1.10:saturation={g['sat']}:gamma={g['gamma']}:brightness={g['bri']}",
        # deep blacks, soft highlight roll-off
        "curves=master='0/0 0.09/0.008 0.25/0.17 0.6/0.58 0.9/0.86 1/0.95'",
        # teal-ish shadows, golden highlights — ties the mixed club lighting together
        "colorbalance=rs=-0.03:gs=0.012:bs=0.02:gm=0.02:rh=0.045:gh=0.02:bh=-0.05",
        "format=yuv420p",
    ]
    return ",".join(vf)

def build_segments(src, work):
    segs = []
    cuts = []
    for i, (b0, b1, clip, tin, _) in enumerate(EDL):
        f0, f1 = fr(b0 * BEAT), fr(b1 * BEAT)
        cuts.append((f0, f1))
        n = f1 - f0
        out = os.path.join(work, f"seg{i:02d}.mp4")
        vf = grade_chain(clip)
        if i == 0:
            vf += f",fade=t=in:st=0:d=0.75"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{tin:.3f}", "-i", os.path.join(src, f"clip{clip}.mov"),
             "-an", "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-preset", "fast", "-crf", "10",
             "-r", str(FPS), out])
        segs.append((f0, f1, out))
    # fill gaps (pre-drop breath) and the reveal with black
    timeline = []
    t = 0
    for f0, f1, p in segs:
        if f0 > t:
            timeline.append(("black", f0 - t))
        timeline.append((p, f1 - f0))
        t = f1
    return timeline, t

def black_seg(work, n, name):
    out = os.path.join(work, name)
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}",
         "-frames:v", str(n), "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p", out])
    return out

# ---------------------------------------------------------------- titles
def ease_out(x):
    x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3

def ease_io(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

def draw_tracked(img, text, font, cy, spacing_px, fill, features=None):
    """Draw centered text with manual letter spacing."""
    d = ImageDraw.Draw(img)
    widths = [d.textlength(ch, font=font, features=features) for ch in text]
    total = sum(widths) + spacing_px * (len(text) - 1)
    x = (img.width - total) / 2
    asc, desc = font.getmetrics()
    y = cy - (asc + desc) / 2
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=font, fill=fill, features=features)
        x += w + spacing_px

def gold_fill(size):
    w, h = size
    y = np.linspace(0, 1, h)[:, None]
    top = np.array([246, 231, 190]); mid = np.array([212, 175, 106]); bot = np.array([140, 104, 52])
    c = np.where(y < 0.5, top + (mid - top) * (y / 0.5), mid + (bot - mid) * ((y - 0.5) / 0.5))
    c = np.repeat(c[:, None, :] if c.ndim == 2 else c, 1, axis=1)
    arr = np.broadcast_to(c.reshape(h, 1, 3), (h, w, 3)).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def render_titles(work, nframes):
    fdir = os.path.join(HERE, "fonts")
    f_small = ImageFont.truetype(os.path.join(fdir, "Inter-Light.otf"), 44)
    f_big = ImageFont.truetype(os.path.join(fdir, "CG-Light.ttf"), 360)
    f_date = ImageFont.truetype(os.path.join(fdir, "Inter-ExtraLight.otf"), 42)
    ivory = (236, 229, 216)
    cy_top, cy_big, cy_rule, cy_date = 720, 940, 1110, 1185

    # pre-render "2027" mask once (large, gold)
    big_mask = Image.new("L", (W, H), 0)
    draw_tracked(big_mask, "2027", f_big, cy_big, 18, 255, features=["lnum"])  # lining figures
    gold = gold_fill((W, H))

    tdir = os.path.join(work, "titles"); os.makedirs(tdir, exist_ok=True)
    for k in range(nframes):
        r = k / FPS
        frame = Image.new("RGB", (W, H), (0, 0, 0))
        # faint warm glow behind the numerals, breathing in
        gl = ease_io((r - 0.5) / 2.0) * 0.55
        if gl > 0:
            yy, xx = np.mgrid[0:H, 0:W]
            d2 = ((xx - W / 2) / 520.0) ** 2 + ((yy - cy_big) / 640.0) ** 2
            glow = np.exp(-d2 * 2.2)[..., None] * np.array([46, 32, 14]) * gl
            frame = Image.fromarray(np.clip(glow, 0, 255).astype(np.uint8), "RGB")

        # NEW YEAR'S EVE — letter spacing settles while fading in
        a = ease_out((r - 0.35) / 0.9)
        if a > 0:
            layer = Image.new("L", (W, H), 0)
            sp = 26 - 8 * a
            draw_tracked(layer, "NEW YEAR’S EVE", f_small, cy_top, sp, 255)
            frame.paste(Image.new("RGB", (W, H), ivory), (0, 0), layer.point(lambda v: int(v * a)))

        # 2027 — blur-to-sharp, slight scale settle
        a = ease_out((r - 0.6) / 1.3)
        if a > 0:
            s = 1.045 - 0.045 * a
            m = big_mask
            if abs(s - 1) > 1e-3:
                sw, sh = int(W * s), int(H * s)
                m = big_mask.resize((sw, sh), Image.LANCZOS)
                ox = (sw - W) // 2; oy = int(cy_big * s - cy_big)
                m = m.crop((ox, oy, ox + W, oy + H))
            blur = 16 * (1 - a)
            if blur > 0.3:
                m = m.filter(ImageFilter.GaussianBlur(blur))
            frame.paste(gold, (0, 0), m.point(lambda v: int(v * a)))
            # shimmer: a soft diagonal highlight sweeping across the numerals
            sh_t = (r - 2.3) / 1.1
            if 0 < sh_t < 1:
                yy, xx = np.mgrid[0:H, 0:W]
                pos = -300 + sh_t * (W + 600)
                band = np.exp(-(((xx + (yy - cy_big) * 0.35) - pos) / 70.0) ** 2)
                band *= np.asarray(m, dtype=np.float32) / 255.0
                arr = np.asarray(frame).astype(np.float32)
                arr += band[..., None] * np.array([70, 60, 40]) * math.sin(math.pi * sh_t)
                frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # hairline rule grows from the centre
        a = ease_out((r - 1.4) / 0.9)
        if a > 0:
            d = ImageDraw.Draw(frame)
            half = 95 * a
            d.line([(W / 2 - half, cy_rule), (W / 2 + half, cy_rule)], fill=(196, 160, 98), width=2)

        # date
        a = ease_out((r - 1.85) / 0.9)
        if a > 0:
            layer = Image.new("L", (W, H), 0)
            draw_tracked(layer, "31.12.2026", f_date, cy_date + 10 * (1 - a), 16, 255)
            frame.paste(Image.new("RGB", (W, H), ivory), (0, 0), layer.point(lambda v: int(v * a)))

        # fade out the last 0.5 s
        tail = (nframes - k) / FPS
        if tail < 0.5:
            frame = Image.eval(frame, lambda v, f=tail / 0.5: int(v * f))
        frame.save(os.path.join(tdir, f"t{k:04d}.png"))
    return tdir

# ---------------------------------------------------------------- audio
def load_audio(path, start, dur):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", path, "-t", f"{dur:.4f}",
           "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    need = int(round(dur * SR))
    if len(a) < need:
        a = np.vstack([a, np.zeros((need - len(a), 2), np.float32)])
    return a[:need]

def build_audio(src, work):
    n_total = int(TOTAL * SR)
    t = np.arange(n_total) / SR
    music = load_audio(os.path.join(src, "clip5.mov"), A0, END_MUSIC + 0.6)
    n_music = int(END_MUSIC * SR)

    # filter sweep over intro/build: muffled -> open, zero-phase so blending never combs
    lp = sosfiltfilt(butter(2, 700, "low", fs=SR, output="sos"), music, axis=0)
    tm = np.arange(len(music)) / SR
    g = np.clip(tm / (DROP - 0.55), 0, 1) ** 2.2
    swept = lp * (1 - g)[:, None] + music * g[:, None]
    # pre-drop breath: dip the gap before the drop
    breath = 1 - 0.55 * np.clip((tm - (DROP - 0.42)) / 0.06, 0, 1) * (tm < DROP)
    swept *= breath[:, None]
    # gentle fade in
    swept *= np.clip(tm / 0.6, 0, 1)[:, None]

    out = np.zeros((n_total, 2), np.float32)
    body = swept[:n_music].copy()
    body[-int(0.006 * SR):] *= np.linspace(1, 0, int(0.006 * SR))[:, None]  # click-free hard stop
    out[:n_music] = body

    # reverb tail from the last two seconds of music, continuing past the stop
    rng = np.random.default_rng(7)
    ir_len = int(3.8 * SR)
    ir_t = np.arange(ir_len) / SR
    ir = rng.standard_normal((ir_len, 2)) * np.exp(-ir_t / 0.9)[:, None]
    ir = sosfiltfilt(butter(2, 3500, "low", fs=SR, output="sos"), ir, axis=0)
    ir[: int(0.02 * SR)] = 0
    seg = swept[n_music - 2 * SR:n_music]
    wet = np.stack([fftconvolve(seg[:, c], ir[:, c]) for c in range(2)], 1)
    wet = wet[2 * SR:]  # only what rings after the stop
    wet /= np.abs(wet).max() + 1e-9
    wet *= 0.22
    L = min(len(wet), n_total - n_music)
    out[n_music:n_music + L] += wet[:L]

    # sub impact on the reveal
    ti = t[n_music:] - t[n_music]
    f = 32 + 26 * np.exp(-ti / 0.35)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = (1 - np.exp(-ti / 0.004)) * np.exp(-ti / 0.75)
    boom = np.sin(phase) * env * 0.55
    air = rng.standard_normal(len(ti)) * np.exp(-ti / 0.25) * 0.06
    air = sosfiltfilt(butter(2, [200, 2500], "band", fs=SR, output="sos"), air)
    out[n_music:, 0] += boom + air
    out[n_music:, 1] += boom + air

    # airy swell as the numerals form (very low)
    r0 = n_music + int(0.4 * SR); r1 = n_music + int(2.6 * SR)
    sw = rng.standard_normal(r1 - r0)
    sw = sosfiltfilt(butter(2, [3000, 9000], "band", fs=SR, output="sos"), sw)
    x = np.linspace(0, 1, r1 - r0)
    sw *= np.sin(np.pi * x) ** 2 * 0.018
    out[r0:r1, 0] += sw; out[r0:r1, 1] += sw[::-1]

    # final fade
    out[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))[:, None]
    raw = os.path.join(work, "mix_raw.f32")
    out.astype(np.float32).tofile(raw)
    wav = os.path.join(work, "mix.wav")
    # loudness for Reels: -14 LUFS integrated, -1 dBTP
    run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", raw,
         "-af", "loudnorm=I=-14:TP=-1.2:LRA=11", "-ar", str(SR), wav])
    return wav

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "NYE2027_teaser.mp4"))
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    timeline, end_f = build_segments(a.src, a.work)
    total_f = fr(TOTAL)
    reveal_f = fr(END_MUSIC)
    assert end_f == reveal_f, (end_f, reveal_f)
    tdir = render_titles(a.work, total_f - reveal_f)

    lst = os.path.join(a.work, "concat.txt")
    with open(lst, "w") as fh:
        for i, (p, n) in enumerate(timeline):
            if p == "black":
                p = black_seg(a.work, n, f"black{i:02d}.mp4")
            fh.write(f"file '{p}'\n")
    edit = os.path.join(a.work, "edit.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", edit])

    wav = build_audio(a.src, a.work)

    # finishing: bloom on highlights, vignette, fine grain -> then title section appended
    finish = (
        "[0:v]split[a][b];"
        "[b]curves=master='0/0 0.75/0 1/1',gblur=sigma=30[g];"
        "[a][g]blend=all_mode=screen:all_opacity=0.18,"
        "vignette=angle=PI/4.6,noise=alls=5:allf=t,setsar=1,format=yuv420p[ed];"
        f"[1:v]fps={FPS},format=yuv420p,noise=alls=3:allf=t,setsar=1[ti];"
        "[ed][ti]concat=n=2:v=1:a=0[v]"
    )
    run(["ffmpeg", "-v", "error", "-y", "-i", edit, "-framerate", str(FPS), "-i", os.path.join(tdir, "t%04d.png"),
         "-i", wav, "-filter_complex", finish, "-map", "[v]", "-map", "2:a",
         "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "14", "-maxrate", "25M",
         "-bufsize", "50M", "-pix_fmt", "yuv420p", "-r", str(FPS), "-color_primaries", "bt709",
         "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-t", f"{TOTAL}", a.out])
    with open(os.path.join(HERE, "edl.json"), "w") as fh:
        json.dump([dict(start=round(b0 * BEAT, 3), end=round(b1 * BEAT, 3), clip=c, src_in=s, note=nt)
                   for b0, b1, c, s, nt in EDL], fh, indent=1)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
