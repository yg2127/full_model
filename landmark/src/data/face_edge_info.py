"""478 FaceMesh landmark 의 polyline-style edge group 정의.

친구 코드의 `_C.{WFLW,COFW,300W}.EDGE_INFO` 형식과 호환:
    [(is_closed: bool, indices: List[int]), ...]

각 entry 의 indices 는 cv2.polylines / spline fit 가 가능하도록 path 순서로 정렬됨.
MediaPipe `FACEMESH_*` connection set 에서 추출.
"""
from __future__ import annotations

from typing import List, Tuple


# ---------- MediaPipe FaceMesh connection sets ----------
# mediapipe.solutions.face_mesh.FACEMESH_* 의 frozenset 을 그대로 사용.
def _load_mp_constants():
    import mediapipe as mp
    fm = mp.solutions.face_mesh
    return {
        "LEFT_EYE":      fm.FACEMESH_LEFT_EYE,
        "RIGHT_EYE":     fm.FACEMESH_RIGHT_EYE,
        "LEFT_EYEBROW":  fm.FACEMESH_LEFT_EYEBROW,
        "RIGHT_EYEBROW": fm.FACEMESH_RIGHT_EYEBROW,
        "LIPS":          fm.FACEMESH_LIPS,
        "FACE_OVAL":     fm.FACEMESH_FACE_OVAL,
        "NOSE":          fm.FACEMESH_NOSE,
        "LEFT_IRIS":     fm.FACEMESH_LEFT_IRIS,
        "RIGHT_IRIS":    fm.FACEMESH_RIGHT_IRIS,
    }


def _edges_to_polylines(edges, allow_open: bool = True) -> List[Tuple[bool, List[int]]]:
    """edge set 을 connected component 별 polyline 으로 분해.

    return: [(is_closed, ordered_indices), ...]  — connected component 마다 1 entry.
    """
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    visited_nodes: set = set()
    polylines: List[Tuple[bool, List[int]]] = []

    while True:
        remaining = [n for n in adj if n not in visited_nodes]
        if not remaining:
            break

        # 시작점: open path 면 degree=1 endpoint, 아니면 가장 작은 idx
        sub_nodes = set()
        stack = [remaining[0]]
        while stack:
            cur = stack.pop()
            if cur in sub_nodes:
                continue
            sub_nodes.add(cur)
            for nb in adj[cur]:
                if nb not in sub_nodes:
                    stack.append(nb)

        endpoints = [n for n in sub_nodes if len(adj[n]) == 1]
        is_closed = len(endpoints) == 0
        start = min(endpoints) if endpoints else min(sub_nodes)

        polyline = [start]
        prev = None
        cur = start
        while True:
            candidates = sorted(n for n in adj[cur] if n != prev)
            # cycle 이 닫히면 stop (closed contour 의 경우)
            if is_closed and start in candidates and len(polyline) > 2:
                break
            nexts = [n for n in candidates if n not in polyline]
            if not nexts:
                break
            nxt = nexts[0]
            polyline.append(nxt)
            prev = cur
            cur = nxt
            if len(polyline) > len(sub_nodes) + 1:
                break

        visited_nodes |= sub_nodes
        polylines.append((is_closed, polyline))

    return polylines


def build_edge_info_478() -> List[Tuple[bool, List[int]]]:
    """MediaPipe FACEMESH_* 에서 478 점의 EDGE_INFO 를 자동 생성.

    각 부위 (eye, brow, lips, face_oval, nose, iris) 가 1+ polyline 으로.
    """
    mp_const = _load_mp_constants()
    # 각 부위는 single closed cycle 인 경우가 대부분.
    parts_closed = ["LEFT_EYE", "RIGHT_EYE", "LEFT_IRIS", "RIGHT_IRIS", "FACE_OVAL"]
    parts_open = ["LEFT_EYEBROW", "RIGHT_EYEBROW"]
    parts_multi = ["LIPS", "NOSE"]  # 외/내, bridge/tip 등 multi-component

    edge_info: List[Tuple[bool, List[int]]] = []
    for part in parts_closed + parts_open + parts_multi:
        edges = mp_const[part]
        polys = _edges_to_polylines(edges)
        edge_info.extend(polys)
    return edge_info


# ---------- 300W 68-point subset mapping ----------
# MediaPipe FaceMesh 478 → dlib/300W 68 point 표준 mapping.
# https://github.com/google-ai-edge/mediapipe/issues/1615 등에서 합의.
MP478_TO_300W68: List[int] = [
    # face contour (0-16, 17 pts)
    127, 234, 132, 172, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 356,
    # left eyebrow (17-21, 5 pts)
    70, 63, 105, 66, 107,
    # right eyebrow (22-26, 5 pts)
    336, 296, 334, 293, 300,
    # nose bridge (27-30, 4 pts)
    168, 197, 5, 4,
    # nose bottom (31-35, 5 pts)
    75, 97, 2, 326, 305,
    # left eye (36-41, 6 pts)
    33, 160, 158, 133, 153, 144,
    # right eye (42-47, 6 pts)
    362, 385, 387, 263, 373, 380,
    # outer mouth (48-59, 12 pts)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # inner mouth (60-67, 8 pts)
    78, 81, 13, 311, 308, 402, 14, 178,
]
assert len(MP478_TO_300W68) == 68


def build_edge_info_68() -> List[Tuple[bool, List[int]]]:
    """300W 68-point convention 의 EDGE_INFO (친구 _C.W300.EDGE_INFO 그대로)."""
    return [
        (False, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        (False, [17, 18, 19, 20, 21]),
        (False, [22, 23, 24, 25, 26]),
        (False, [27, 28, 29, 30]),
        (False, [31, 32, 33, 34, 35]),
        (False, [36, 37, 38, 39]),
        (False, [39, 40, 41, 36]),
        (False, [42, 43, 44, 45]),
        (False, [45, 46, 47, 42]),
        (False, [48, 49, 50, 51, 52, 53, 54]),
        (False, [54, 55, 56, 57, 58, 59, 48]),
        (False, [60, 61, 62, 63, 64]),
        (False, [64, 65, 66, 67, 60]),
    ]


# ---------- 478 inter-eye anchor (NME normalize 용) ----------
# MediaPipe 의 outer eye corner — 33 (left), 263 (right).
NME_ANCHOR_478: Tuple[int, int] = (33, 263)
# 300W 68 의 outer eye — 36 (left), 45 (right).
NME_ANCHOR_68: Tuple[int, int] = (36, 45)


if __name__ == "__main__":
    info = build_edge_info_478()
    print(f"478 EDGE_INFO: {len(info)} polylines")
    for is_closed, idxs in info:
        print(f"  closed={is_closed}  len={len(idxs)}  first={idxs[:5]} ... last={idxs[-3:]}")
    print()
    info68 = build_edge_info_68()
    print(f"68 EDGE_INFO: {len(info68)} polylines (300W convention)")
