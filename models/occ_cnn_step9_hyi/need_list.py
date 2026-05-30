# list_needed_face_files_with_size.py
from pathlib import Path
import json

MANIFEST_PATHS = [
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_train.json"),
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_val.json"),
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_test_clean.json"),
    Path("/home/hyi/Code/Step9_extract_crop_npz/clip_manifest_test_masked.json"),
]

CLEAN_FACEMESH_ROOT = Path("/data/shared/DMD_landmarks/facemesh")
CLEAN_VIDEO_ROOT = Path("/data/shared/DMD")

MASKED_FACEMESH_ROOT = Path(
    "/data/shared/Occlusion_subset_dataset/"
    "region_occlusion_video_dataset_v3_original_fixedmask_yolo_face_facemesh_canonical/facemesh"
)

MASKED_VIDEO_CANONICAL_ROOT = Path(
    "/data/shared/Occlusion_subset_dataset/"
    "region_occlusion_video_dataset_v3_original_fixedmask/videos_canonical"
)

OUT_TXT = Path("/home/hyi/Code/Step9_extract_crop_npz/needed_face_files.txt")
OUT_JSON = Path("/home/hyi/Code/Step9_extract_crop_npz/needed_face_files_size_summary.json")
OUT_SUMMARY_TXT = Path("/home/hyi/Code/Step9_extract_crop_npz/needed_face_files_size_summary.txt")


def load_json_or_jsonl(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(num_bytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def file_size(path: Path) -> int:
    try:
        if path.exists() or path.is_symlink():
            return path.stat().st_size
    except Exception:
        pass
    return 0


def clean_face_npz_to_video(face_npz: Path) -> Path:
    rel = face_npz.relative_to(CLEAN_FACEMESH_ROOT)
    video_dir = CLEAN_VIDEO_ROOT / rel.parent

    name = face_npz.name
    candidates = []

    if name.endswith("_ir_face_facemesh.npz"):
        base = name.replace("_ir_face_facemesh.npz", "")
        candidates.extend([
            video_dir / f"{base}_ir_face.mp4",
            video_dir / f"{base}_ir_body.mp4",
            video_dir / f"{base}_rgb_body.mp4",
        ])

    stem = name.replace("_ir_face_facemesh.npz", "")
    if video_dir.exists():
        candidates.extend(sorted(video_dir.glob(f"{stem}*.mp4")))

    for p in candidates:
        if p.exists() or p.is_symlink():
            return p

    raise FileNotFoundError(f"clean video not found for {face_npz}")


def masked_face_npz_to_video(face_npz: Path) -> Path:
    rel = face_npz.relative_to(MASKED_FACEMESH_ROOT)
    video_name = face_npz.name.replace("_ir_face_facemesh.npz", "_ir_face.mp4")
    video = MASKED_VIDEO_CANONICAL_ROOT / rel.parent / video_name

    if video.exists() or video.is_symlink():
        return video

    raise FileNotFoundError(f"masked video not found for {face_npz}")


def add_file(
    files: set,
    category_files: dict,
    path: Path,
    category: str,
):
    files.add(path)
    category_files.setdefault(category, set()).add(path)


def main():
    needed_npz = set()

    manifest_clip_counts = {}

    for mp in MANIFEST_PATHS:
        rows = load_json_or_jsonl(mp)
        manifest_clip_counts[str(mp)] = len(rows)
        print(mp, len(rows))

        for r in rows:
            if "face_npz" in r:
                needed_npz.add(Path(r["face_npz"]))

    files = set()
    category_files = {}

    # manifests
    for mp in MANIFEST_PATHS:
        add_file(files, category_files, mp, "manifest")

    missing = []

    clean_npz_count = 0
    masked_npz_count = 0
    clean_video_count = 0
    masked_video_count = 0

    for face_npz in sorted(needed_npz):
        try:
            if str(face_npz).startswith(str(CLEAN_FACEMESH_ROOT)):
                clean_npz_count += 1
                add_file(files, category_files, face_npz, "clean_facemesh_npz")

                video = clean_face_npz_to_video(face_npz)
                clean_video_count += 1
                add_file(files, category_files, video, "clean_video")

            elif str(face_npz).startswith(str(MASKED_FACEMESH_ROOT)):
                masked_npz_count += 1
                add_file(files, category_files, face_npz, "masked_facemesh_npz")

                video = masked_face_npz_to_video(face_npz)
                masked_video_count += 1
                add_file(files, category_files, video, "masked_video")

            else:
                raise ValueError(f"unknown root: {face_npz}")

        except Exception as e:
            missing.append((str(face_npz), str(e)))

    # file list 저장
    OUT_TXT.write_text(
        "\n".join(str(p) for p in sorted(files)) + "\n",
        encoding="utf-8"
    )

    # category별 용량 계산
    category_summary = {}

    for category, paths in category_files.items():
        total_size = sum(file_size(p) for p in paths)
        category_summary[category] = {
            "num_files": len(paths),
            "total_bytes": total_size,
            "human_size": human_size(total_size),
        }

    total_bytes = sum(file_size(p) for p in files)

    summary = {
        "unique_face_npz": len(needed_npz),
        "total_files_to_copy": len(files),
        "missing": len(missing),

        "clean_npz_count": clean_npz_count,
        "masked_npz_count": masked_npz_count,
        "clean_video_count": clean_video_count,
        "masked_video_count": masked_video_count,

        "total_bytes": total_bytes,
        "total_human_size": human_size(total_bytes),

        "manifest_clip_counts": manifest_clip_counts,
        "category_summary": category_summary,

        "out_txt": str(OUT_TXT),
        "out_json": str(OUT_JSON),
        "out_summary_txt": str(OUT_SUMMARY_TXT),
    }

    OUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    lines = []
    lines.append("=" * 80)
    lines.append("NEEDED FACE FILES SIZE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"unique face_npz       : {len(needed_npz)}")
    lines.append(f"total files to copy   : {len(files)}")
    lines.append(f"missing               : {len(missing)}")
    lines.append("")
    lines.append(f"TOTAL SIZE            : {human_size(total_bytes)}")
    lines.append(f"TOTAL BYTES           : {total_bytes}")
    lines.append("")
    lines.append("[Counts]")
    lines.append(f"clean npz             : {clean_npz_count}")
    lines.append(f"masked npz            : {masked_npz_count}")
    lines.append(f"clean video           : {clean_video_count}")
    lines.append(f"masked video          : {masked_video_count}")
    lines.append("")
    lines.append("[Category sizes]")

    for category, info in sorted(category_summary.items()):
        lines.append(
            f"{category:22s} "
            f"files={info['num_files']:6d} "
            f"size={info['human_size']:>12s}"
        )

    lines.append("")
    lines.append(f"saved file list       : {OUT_TXT}")
    lines.append(f"saved json summary    : {OUT_JSON}")

    OUT_SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))

    if missing:
        miss_path = OUT_TXT.with_name("needed_face_files_missing.txt")
        miss_path.write_text(
            "\n".join(f"{p}\t{e}" for p, e in missing),
            encoding="utf-8"
        )
        print("missing saved:", miss_path)


if __name__ == "__main__":
    main()