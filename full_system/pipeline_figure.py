from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = Path(__file__).resolve().parent


COLORS = {
    "input": "#f7f7f7",
    "det": "#eaf3fb",
    "occ": "#f7f1e4",
    "proc": "#eef4ee",
    "model": "#edf4ff",
    "output": "#f4f4f4",
}


def box(ax, xy, w, h, text, fc, fontsize=8.6, lw=1.0):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor="#2f2f2f",
        facecolor=fc,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        family="DejaVu Sans",
        linespacing=1.18,
    )
    return patch


def right(p):
    return p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2


def left(p):
    return p.get_x(), p.get_y() + p.get_height() / 2


def top(p):
    return p.get_x() + p.get_width() / 2, p.get_y() + p.get_height()


def bottom(p):
    return p.get_x() + p.get_width() / 2, p.get_y()


def arrow(ax, start, end, text=None, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=0.9,
        color="#333333",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.075,
            text,
            ha="center",
            va="center",
            fontsize=7.0,
            color="#333333",
            family="DejaVu Sans",
        )


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(18.2, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    # Inputs
    face_in = box(ax, (0.35, 3.95), 1.35, 0.72, "Face frame\nBGR image", COLORS["input"])
    body_in = box(ax, (0.35, 1.25), 1.35, 0.72, "Body frame\nBGR image", COLORS["input"])

    # Face branch
    face_bbox = box(ax, (2.10, 3.95), 1.45, 0.72, "YOLO-Face\nbbox + score", COLORS["det"])
    facemesh = box(ax, (4.05, 4.62), 1.70, 0.76, "MediaPipe\nFaceMesh\n478 x 3", COLORS["det"], fontsize=8.2)
    occ = box(ax, (4.05, 3.42), 1.70, 0.76, "Occ CNN\nvisibility + valid\n5-D feature", COLORS["occ"], fontsize=8.2)
    restore = box(ax, (6.35, 4.62), 1.72, 0.76, "ORFormer + HGNet\nconditional\nrestoration", COLORS["occ"], fontsize=8.0)
    merge = box(ax, (8.55, 3.95), 1.85, 0.82, "Occ-gated\nFaceMesh merge\nraw or restored", COLORS["proc"], fontsize=8.1)

    # Body branch
    pose = box(ax, (2.10, 1.25), 1.45, 0.72, "YOLO-Pose\nCOCO17 skeleton", COLORS["det"])
    pose_pre = box(ax, (4.05, 1.25), 1.90, 0.72, "Pose preprocessing\nxy + bone + vel + conf", COLORS["proc"], fontsize=7.9)

    # Temporal and classifier
    buffer = box(ax, (11.05, 2.62), 1.55, 0.88, "Temporal buffer\nT = 48 frames", COLORS["proc"])
    preproc = box(ax, (13.08, 2.62), 1.55, 0.88, "DMS preprocessing\npose, face, occ", COLORS["proc"])
    clf = box(
        ax,
        (15.08, 2.52),
        2.60,
        1.08,
        "Model4 DMS classifier\nPoseBranch + FaceBranch\nexplicit region-scalar mask gate",
        COLORS["model"],
        fontsize=7.8,
        lw=1.2,
    )
    heads = box(ax, (15.90, 0.88), 0.95, 0.98, "Heads\naction\ngaze\nhands\ntalk", COLORS["model"], fontsize=7.6, lw=1.2)
    out = box(ax, (17.05, 0.88), 0.98, 0.98, "Prediction\nJSONL\nprob/conf", COLORS["output"], fontsize=7.7)

    # Arrows: face stream
    arrow(ax, right(face_in), left(face_bbox))
    arrow(ax, right(face_bbox), left(facemesh), rad=0.10)
    arrow(ax, right(face_bbox), left(occ), rad=-0.05)
    arrow(ax, right(occ), left(restore), text="if occluded", rad=0.18)
    arrow(ax, right(facemesh), left(merge), rad=-0.06)
    arrow(ax, right(restore), (8.55, 4.50), rad=0.02)
    arrow(ax, right(merge), (11.05, 3.28), rad=-0.07)
    arrow(ax, (5.75, 3.80), (11.05, 3.05), rad=-0.10)

    # Arrows: body stream
    arrow(ax, right(body_in), left(pose))
    arrow(ax, right(pose), left(pose_pre))
    arrow(ax, right(pose_pre), (11.05, 2.95), rad=0.08)

    # Classifier and output
    arrow(ax, right(buffer), left(preproc))
    arrow(ax, right(preproc), left(clf))
    arrow(ax, bottom(clf), top(heads), rad=0.05)
    arrow(ax, right(heads), left(out))

    # Subtle branch labels
    ax.text(0.35, 5.38, "Face stream", fontsize=9.2, weight="bold", family="DejaVu Sans", color="#222222")
    ax.text(0.35, 2.72, "Body stream", fontsize=9.2, weight="bold", family="DejaVu Sans", color="#222222")
    ax.text(11.05, 4.03, "Windowed multi-modal inference", fontsize=9.2, weight="bold", family="DejaVu Sans", color="#222222")

    # Minimal note in paper-figure style.
    ax.text(
        0.35,
        0.22,
        "Failure policy: missing body/face detections are replaced by zero skeleton/FaceMesh and neutral occlusion features; inference continues once the 48-frame buffer is ready.",
        fontsize=7.1,
        family="DejaVu Sans",
        color="#444444",
    )

    fig.savefig(OUT_DIR / "pipeline_figure.png", bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(OUT_DIR / "pipeline_figure.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_pipeline()
