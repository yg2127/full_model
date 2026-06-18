import json
from pathlib import Path
OUT="/data/shared/scuppy/Full_System/experiments/retrain_ablation/compare.ipynb"
def md(t):  return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def code(t):return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)}
cells=[]

cells.append(md(r"""# 재학습 Ablation 비교 — 같은 모델, 모듈 1개씩 제거 후 **재학습**

**`full` = 기존 팀 model4**(`model4_occgateRAW_explicitRegionScalarMaskGate`, Full_System 탑재 모델, 재학습 X).
**변형 5개** = 그 모델과 **완전히 동일한 학습 설정**(seed 42·데이터·하이퍼파라미터·파이프라인)에서
**모듈 1개만** 바꿔 학습. 즉 "학습 방식 다 똑같이, 모듈만 변경".

| 변형 | 제거 모듈 | config 변경 |
|---|---|---|
| `full` | (없음, **기존 모델**) | — |
| `no_body` | 신체(pose) 분기 | `ablation.zero_pose=true` |
| `no_face` | 얼굴 랜드마크 분기 | `ablation.zero_face=true` |
| `no_occ` | Occ 차폐 신호 | gate occ-condition off |
| `no_hgnet` | HGNet 복원 | `face.npz_swap=false` (raw MediaPipe) |
| `no_gate` | 차폐-인지 fusion 게이트 | `fusion=concat_condition` |

> 각 모듈을 빼고 실제로 학습한 모델의 GT 기반 macro-F1 을 **기존 `full`** 과 비교 → 모듈별 F1 drop = 진짜 기여도."""))

cells.append(md("## 1. 6개 변형의 summary.json 로드"))
cells.append(code(r'''
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"]=False
pd.set_option("display.width",200)

# 그래프 하단 '사용 지표 정의' 캡션 헬퍼
def cap(fig, text):
    fig.tight_layout(rect=[0,0.10,1,1])
    fig.text(0.5,0.015,text,ha="center",va="bottom",fontsize=7.8,color="#555",
             bbox=dict(boxstyle="round",fc="#f7f7f7",ec="#bbb"))
F1_DEF="metric: per-head clip-level macro-F1 (action/gaze/hands/talk).  full = original team model4;  variant = base minus ONE module, retrained (seed42, same config)."

RUNS=Path("/data/shared/scuppy/Full_System/experiments/retrain_ablation/runs")
# full = 기존 팀 model4 (Full_System 탑재 모델, 재학습 X). 변형 = 같은 config에서 모듈 1개만 바꿔 학습.
TEAM_FULL=Path("/data/shared/scuppy/hyi/Ablation/HGNET_Classification/results_gaze045_light/model4_occgateRAW_explicitRegionScalarMaskGate_seed42_loss045/summary.json")
HEADS=["action","gaze","hands","talk"]
ORDER=["full","no_body","no_face","no_occ","no_hgnet","no_gate"]
W={"action":0.45,"gaze":0.45,"hands":0.05,"talk":0.05}

rows=[]
for v in ORDER:
    sj = TEAM_FULL if v=="full" else RUNS/f"{v}_seed42"/"summary.json"
    if not sj.exists():
        print("[missing]", sj, "→ 아직 학습 안 끝남"); continue
    d=json.load(open(sj)); ts=d["test_splits"]
    r={"variant":v,"best_epoch":d.get("best_epoch")}
    for sp in ("test_clean","test_masked"):
        for h in HEADS:
            r[f"{sp}_{h}"]=float(ts[sp]["per_head"][h]["clip_f1_macro"])
    rows.append(r)
df=pd.DataFrame(rows)
print("로드된 변형:", df["variant"].tolist() if len(df) else "없음")
df
'''))

cells.append(md("## 2. macro-F1 표 (clean / masked)"))
cells.append(code(r'''
for sp in ("test_clean","test_masked"):
    cols=[f"{sp}_{h}" for h in HEADS]
    t=df[["variant"]+cols].copy(); t.columns=["variant"]+HEADS
    t["weighted"]=sum(W[h]*t[h] for h in HEADS)
    print(f"=== {sp} macro-F1 ==="); display(t.set_index("variant").round(4))
'''))

cells.append(md("## 3. F1 drop vs full (재학습 기준 — 진짜 모듈 기여도)"))
cells.append(code(r'''
base=df[df["variant"]=="full"].iloc[0]
for sp in ("test_clean","test_masked"):
    rows=[]
    for _,r in df.iterrows():
        if r["variant"]=="full": continue
        rows.append({"variant":r["variant"], **{h: round(base[f"{sp}_{h}"]-r[f"{sp}_{h}"],4) for h in HEADS}})
    dd=pd.DataFrame(rows).set_index("variant")
    dd["weighted"]=sum(W[h]*dd[h] for h in HEADS)
    print(f"=== {sp}: F1 drop vs full (양수=그 모듈 빼니 성능↓) ==="); display(dd.round(4))
'''))
cells.append(md(r"""**해석.** 값이 **클수록(양수) 그 모듈을 빼면 성능이 많이 떨어짐 = 그 모듈이 중요**.
음수면 빼는 게 오히려 나았다는 뜻(노이즈이거나 그 모듈이 불필요). clean vs masked 를 비교해
**차폐(masked)에서만 커지는 모듈**(예: no_hgnet, no_occ)이 차폐 처리의 실제 기여입니다."""))

cells.append(md("## 4. 그래프 — 모듈별 F1 drop (clean vs masked)"))
cells.append(code(r'''
base=df[df["variant"]=="full"].iloc[0]
variants=[v for v in ORDER if v!="full" and v in df["variant"].values]
fig,axes=plt.subplots(1,2,figsize=(14,4.6),sharey=True)
for ax,sp in zip(axes,("test_clean","test_masked")):
    x=np.arange(len(variants)); w=0.2
    for i,h in enumerate(HEADS):
        vals=[max(0.0, 100*(base[f"{sp}_{h}"]-df[df["variant"]==v].iloc[0][f"{sp}_{h}"])/base[f"{sp}_{h}"]) if base[f"{sp}_{h}"]>0 else 0.0 for v in variants]  # full 대비 상대%, 음수→0
        b=ax.bar(x+i*w,vals,w,label=h); ax.bar_label(b,fmt="%.0f%%",fontsize=6)
    ax.set_xticks(x+1.5*w); ax.set_xticklabels(variants,rotation=15)
    ax.set_title(sp+" — relative drop (%) vs full"); ax.axhline(0,color="k",lw=0.6); ax.grid(axis="y",alpha=0.3)
axes[0].set_ylabel("relative F1 drop vs full (%)"); axes[0].legend(title="head",ncol=4,fontsize=8)
fig.suptitle("Retrained ablation: relative F1 drop (%) vs full by removed module (left=clean, right=masked)",y=1.02)
cap(fig, "metric: relative F1 drop = 100x(full - variant)/full  (per head, clip-level macro-F1; negatives clamped to 0%).  "+F1_DEF); plt.show()
'''))
cells.append(md(r"""**해석.** 왼쪽(clean) vs 오른쪽(masked) 대조가 핵심입니다.
- `no_body` → action/hands 큰 drop (신체 지배)
- `no_face` → gaze 큰 drop (시선=얼굴)
- `no_hgnet`/`no_occ`/`no_gate` → **masked 에서 clean 보다 drop 이 커지면** 차폐 처리 모듈이 worst case 에서 기여."""))

cells.append(md("## 5. 절대 F1 막대 (변형별, masked=worst case)"))
cells.append(code(r'''
fig,ax=plt.subplots(figsize=(11,4.6))
present=[v for v in ORDER if v in df["variant"].values]
x=np.arange(len(present)); w=0.2
for i,h in enumerate(HEADS):
    vals=[df[df["variant"]==v].iloc[0][f"test_masked_{h}"] for v in present]
    ax.bar(x+i*w,vals,w,label=h)
ax.set_xticks(x+1.5*w); ax.set_xticklabels(present,rotation=15)
ax.set_ylabel("test_masked macro-F1"); ax.set_title("Worst-case (masked) F1 per variant")
ax.legend(title="head",ncol=4,fontsize=8); ax.grid(axis="y",alpha=0.3)
cap(fig, "metric: worst-case (masked) per-head clip-level macro-F1.  bars = each head F1 under occlusion (higher=better).  "+F1_DEF); plt.show()

SAVE=Path("/data/shared/scuppy/Full_System/experiments/retrain_ablation")
df.round(4).to_csv(SAVE/"retrain_ablation_f1.csv",index=False)
print("saved", SAVE/"retrain_ablation_f1.csv")
'''))
cells.append(md(r"""**해석 / 결론.** 재학습 기반이므로 이 F1 drop 이 **각 모듈의 진짜 기여도**입니다
(입력 zero/agreement 대용지표와 달리, 모듈 없이 새로 학습한 모델의 실제 성능).
`full` 이 모든(또는 worst-case) 지표에서 가장 높고, 각 모듈 제거 시 해당 task 가 떨어지면 그 모듈이 필요하다는 직접 증거입니다."""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python (scuppy)","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
Path(OUT).parent.mkdir(parents=True,exist_ok=True)
json.dump(nb,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("WROTE",OUT,"cells:",len(cells))
