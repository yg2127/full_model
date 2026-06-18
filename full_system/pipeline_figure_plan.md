# Full_System 논문형 파이프라인 figure 설계 보고서

## 1. 프로젝트 개요

`/data/shared/scuppy/Full_System`는 face/body 두 비디오 스트림을 입력으로 받아 운전자 상태를 추론하는 통합 DMS 런타임이다. 코드 기준 최종 흐름은 YOLO-Pose 기반 body skeleton, YOLO-face 기반 face ROI, MediaPipe FaceMesh, occlusion visibility CNN, 선택적 ORFormer+HGNet landmark restoration, occlusion-gated landmark merge, 48-frame temporal buffer, 그리고 기존 Model4 DMS multitask classifier로 구성된다.

Confirmed:
- Body branch: YOLO-Pose -> COCO17 skeleton.
- Face branch: YOLO-face bbox를 단일 face coordinate basis로 사용.
- FaceMesh: YOLO bbox ROI 위에서 MediaPipe FaceMesh를 실행해 478 landmarks를 생성.
- Occ CNN: YOLO bbox crop에서 `[left_eye, right_eye, nose, mouth]` visibility probability와 crop-valid flag를 생성.
- Restoration: occlusion이 감지된 경우 ORFormer+HGNet을 호출하고, occluded region만 HGNet 좌표로 교체.
- Temporal inference: 최근 48 frames를 DMS classifier에 입력.
- Output heads: `action`, `gaze`, `hands`, `talk`.

Inferred:
- 논문 figure에서는 ORFormer와 HGNet을 하나의 "Optional Landmark Restoration" 모듈로 묶는 것이 가장 가독성이 좋다.
- DMS classifier 내부의 pose/face backbone 및 fusion은 하나의 "Model4 DMS Classifier" 큰 박스로 표현하고, 내부 키워드로 `PoseBranch`, `FaceBranch`, `explicit region-scalar mask gate`, `multi-task heads`를 넣는 것이 적절하다.

## 2. 실행 진입점

Primary entrypoint:
- `scripts/run_video_pair.py`
  - CLI arguments: `--config`, `--face-video`, `--body-video`, `--out-jsonl`, `--max-frames`
  - face/body video를 `cv2.VideoCapture`로 동시에 읽고, 매 frame마다 `FullDMSSystem.step(face_frame, body_frame)` 호출.
  - temporal buffer가 채워진 후 JSONL prediction을 저장.

Demo/overlay entrypoints:
- `scripts/run_hardcoded_overlay.py`
  - hardcoded sample video pair에 대해 overlay video와 prediction JSONL 생성.
- `scripts/run_full_system_warning_overlay_edge_tts_hardcoded_v2.py`
  - FullDMSSystem 위에 warning logic, visual overlay, JSONL event logging, edge-tts audio warning을 추가한 데모 스크립트.

Support entrypoint:
- `scripts/smoke_test_shapes.py`
  - 모델 로딩 없이 buffer shape smoke test 용도.

## 3. 최종 모델 파이프라인 요약

1. Input loading
   - `face_video`와 `body_video`를 OpenCV `VideoCapture`로 열고 frame pair를 순차적으로 읽는다.
   - 각 step 입력은 OpenCV BGR image인 `face_frame`, `body_frame`이다.

2. Body branch
   - `YoloPoseSkeletonExtractor`가 body frame에서 YOLO-Pose 추론을 수행한다.
   - 출력은 COCO17 keypoints `(17, 2)`와 confidence `(17,)`.
   - detection 실패 시 zero skeleton과 zero confidence로 fallback한다.

3. Face detection and face-side coordinate basis
   - `YoloFaceBBoxExtractor`가 face frame에서 face bbox `(x1, y1, x2, y2)`와 detection score를 추출한다.
   - face bbox 실패 시 zero FaceMesh와 neutral occlusion vector로 fallback하고, 시스템은 DMS inference를 계속 진행한다.

4. FaceMesh extraction
   - `MediaPipeFaceMeshOnYoloCrop`가 YOLO bbox ROI에서 MediaPipe FaceMesh를 실행한다.
   - 출력은 `(478, 3)` landmark.
   - FaceMesh 실패 시 zero FaceMesh를 사용한다.

5. Occlusion visibility estimation
   - `OccCNNRealtimeWrapper`가 YOLO bbox crop을 grayscale crop으로 변환해 visibility model에 넣는다.
   - 출력은 four-region visibility probability `(4,)`와 crop-valid flag.
   - DMS classifier로 전달되는 occlusion feature는 `(5,) = [left_eye, right_eye, nose, mouth, crop_valid]`.

6. Conditional landmark restoration
   - `OccGatedFaceMeshMerger.visible_probs_to_labels()`가 visibility threshold보다 낮은 region을 occluded로 표시한다.
   - occlusion이 있을 때만 `HGNetRestorer`가 ORFormer reference heatmap과 HGNet을 사용해 478 landmark를 복원한다.
   - `OccGatedFaceMeshMerger.merge()`는 visible region은 MediaPipe raw landmark를 유지하고, occluded region만 HGNet-restored landmark로 교체한다.

7. Temporal buffer
   - `TemporalDMSBuffer`가 body skeleton, face landmarks, face bbox metadata, face detection flags, occ feature를 frame 단위로 누적한다.
   - window size는 config 기준 `48`.
   - buffer가 준비되기 전에는 prediction을 반환하지 않는다.

8. DMS classifier preprocessing
   - `DMSClassifierWrapper.predict_window()`가 48-frame arrays를 classifier 입력 tensor로 변환한다.
   - Pose preprocessing: center-scale normalization, bone feature, velocity feature, confidence channel. Config 기준 pose input channel은 `2 + 2 + 2 + 1 = 7`.
   - Face preprocessing: bbox-based face-local normalization, z coordinate 사용, detected channel 추가. Config 기준 `facemesh_full` mode이므로 raw 478 landmarks를 모델에 전달하고, model 내부에서 region pooling을 수행한다.
   - Occlusion preprocessing: window-level mean으로 `(5,)` occlusion feature를 생성한다.

9. Model4 DMS classifier
   - `build_model()`이 `MultitaskClassifier`를 생성한다.
   - Body stream: `PoseBranch`.
   - Face stream: `FaceRegionPool` + `FaceBranch`.
   - Joint stream: `ConcatJointFusion` 이후 TGC post blocks, temporal module. Config 기준 temporal kind는 `identity`.
   - Task fusion: config 기준 `explicit_region_scalar_mask_gate`.
   - Heads: action, gaze, hands, talk linear heads.

10. Prediction and postprocessing
   - 각 head logits에 softmax를 적용하고 argmax class id, probability vector, confidence를 반환한다.
   - 출력 JSONL에는 frame index, face status, occlusion labels, occ feature, body detection flag, HGNet 사용 여부, restored regions, debug metadata가 포함된다.

## 4. Figure에 들어갈 주요 모듈

Main figure boxes:
- Input pair
  - `Face frame`
  - `Body frame`
- Body branch
  - `YOLO-Pose`
  - `COCO17 skeleton`
- Face branch
  - `YOLO-Face bbox`
  - `MediaPipe FaceMesh`
  - `Occ CNN`
  - `ORFormer + HGNet`
  - `Occ-gated merge`
- Temporal aggregation
  - `48-frame buffer`
  - `Pose / Face / Occ preprocessing`
- Classifier
  - `Model4 DMS classifier`
  - `PoseBranch + FaceBranch`
  - `Explicit region-scalar mask gate`
- Outputs
  - `Action`
  - `Gaze`
  - `Hands`
  - `Talk`
  - `JSONL / warning metadata`

Small helper text:
- Body skeleton: `(17, 2) + conf`
- FaceMesh: `(478, 3)`
- Occ feature: `(5,)`
- Window: `T=48`
- Heads: `11 / 9 / 4 / 2 classes` where these class counts are based on the configured project label space used in prior DMD experiments. The code obtains gaze count from `NUM_GAZE_ZONES`.

## 5. Confirmed vs Inferred

Confirmed from code:
- `scripts/run_video_pair.py` is the cleanest runtime CLI entrypoint.
- `FullDMSSystem.step()` processes one face/body frame pair at a time.
- YOLO-Pose, YOLO-Face, MediaPipe FaceMesh, Occ CNN, HGNetRestorer, OccGatedFaceMeshMerger, TemporalDMSBuffer, and DMSClassifierWrapper are explicitly instantiated in `FullDMSSystem.__init__`.
- The temporal buffer appends body, face, bbox, detection flags, and occ feature before invoking the DMS classifier.
- `DMSClassifierWrapper` loads the classifier config and checkpoint, calls `build_model()`, loads checkpoint state dict with `strict=True`, then applies softmax and argmax per head.
- Runtime config paths point to:
  - `Model/yolo_pose.pt`
  - `Model/yolo_face.pt`
  - `Model/occ_cnn.pt`
  - `Model/orformer.pt`
  - `Model/hgnet.pt`
  - `Model/dms_checkpoint.pt`
- DMS config uses `model.fusion.kind=explicit_region_scalar_mask_gate`, `window.size=48`, `face.mode=facemesh_full`, `face.num_landmarks=478`, and `occ.dim=5`.

Inferred or simplified for figure:
- The paper figure groups ORFormer and HGNet into one restoration block to avoid too much architectural detail.
- The DMS classifier internal TGC blocks, temporal identity module, and task fusion details are summarized inside one classifier module.
- Warning overlay/TTS logic is excluded from the main model figure because it is demo-level postprocessing, not core model inference.

## 6. Figure 레이아웃 설명

Recommended layout:
- Horizontal left-to-right flow.
- Top branch: face stream.
- Bottom branch: body stream.
- Face branch splits after `YOLO-Face bbox` into:
  - MediaPipe FaceMesh branch.
  - Occ CNN branch.
  - Conditional ORFormer+HGNet restoration branch, triggered by occlusion labels.
- These face-side branches converge at `Occ-gated FaceMesh`.
- Body skeleton and final FaceMesh/Occ feature converge at `48-frame buffer`.
- The classifier is shown as one large module after preprocessing.
- Final outputs are four compact boxes aligned vertically or grouped into a single output box.

Visual style:
- White background.
- Thin dark gray borders.
- Light gray for input/preprocessing boxes.
- Light blue for detection/feature extraction.
- Light beige for occlusion/restoration.
- Light green for classifier/output.
- No decorative gradients or heavy colors.

## 7. 생성된 파일 목록

The following files are generated in `/data/shared/scuppy/Full_System`:

- `pipeline_figure_plan.md`
- `pipeline_figure.mmd`
- `pipeline_figure.dot`
- `pipeline_figure.py`
- `pipeline_figure.png`
- `pipeline_figure.svg`
- `figure_caption.txt`
