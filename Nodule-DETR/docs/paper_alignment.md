# Paper-to-code correspondence

Reference: the author's accepted revised manuscript, sections 3.1–3.5 and 4.2.1, including architecture Figures 1 and 4. This document describes implementation alignment, not a new experimental reproduction.

## Architecture

| Manuscript component | Implementation | Checked behavior |
|---|---|---|
| MSFCA, §3.2 / Figure 1 | `models/nodule_detr/resnet_msfca.py`, `msfca.py` | ResNet50 stage depths 3/4/6/3; attention in all 16 bottlenecks, after the residual branch's last BatchNorm and before shortcut addition, as in Figure 1. Fixed grouped 2D DCT, concatenated frequency descriptors, learned channel gates. |
| HFF, §3.3 / Figure 4 | `model.py`, `hff.py` | Stage 2/3/4 outputs have strides 8/16/32. 1×1 projections generate P1–P3; 3×3 stride-2 convolution and GroupNorm generate P4 from stage 4. Bottom-up additive SCDown fusion produces F1–F4. |
| SCDown | `hff.py` | Pointwise 1×1 channel mixing followed by 3×3 depthwise spatial convolution; stride 2 in HFF. |
| MSDA, §3.4 | `deformable_transformer.py`, `msda.py` | Encoder self-attention and decoder cross-attention use deformable attention over 4 levels, 8 heads, 4 points per level/head. Weights are jointly normalized across level/point dimensions. Decoder query self-attention remains ordinary multi-head attention, matching Figure 1. |
| Prediction / assignment / loss, §3.5 | `model.py`, `matcher.py`, `losses.py`, `denoising.py` | Hungarian assignment; sigmoid focal loss (gamma 2), L1 and GIoU; weights 2/5/2, including the denoising branch. |

HFF follows the supplied full-model forward computation:

```text
F1 = P1
F2 = P2 + resize(D1(F1), spatial_size(P2))
F3 = P3 + resize(D2(F2), spatial_size(P3))
F4 = P4 + resize(D2(F3), spatial_size(P4))
```

The source shares D2 between the final two transitions. It also registers an unused `hff.scDown3`; this is retained to avoid silently changing the historical parameter set. This unused module receives no gradient. Resizing is nearest-neighbor alignment for rounding differences at odd spatial sizes.

The supplied backbone additionally uses SCDown on dimension-changing residual shortcuts. That source detail and HFF weight sharing are not spelled out in the manuscript's high-level description; they are preserved and disclosed here. MSFCA uses the source's top16 frequency selection, reduction factor 16, and reference DCT spatial sizes 56/28/14/7. SCDown includes the source's BatchNorm and SiLU after its pointwise convolution, and BatchNorm without activation after its depthwise convolution. These are implementation details, not newly tested architectural variants.

## Configured defaults (see unresolved findings below)

| Setting | Value | Where enforced |
|---|---|---|
| Encoder / decoder depth | 6 / 6 | `main.py`, transformer builder |
| Hidden width / heads | 256 / 8 | `main.py`, attention modules |
| Feature levels / encoder points / decoder points | 4 / 4 / 4 | `main.py`, attention modules |
| Object queries | 300 | `main.py`, detector |
| Feature-map sine encoding base, both axes | 20; not equivalent to the manuscript temperature formula | `position_encoding.py` |
| Epochs / batch size | 200 / 2 | `main.py` |
| Optimizer / weight decay | AdamW / 1e-4 | `main.py` |
| Non-backbone / backbone LR | 1e-4 / 1e-5 | optimizer parameter groups |
| LR schedule | StepLR, 40 epochs, gamma 0.1 | `main.py` |
| Focal / L1 / GIoU coefficients | 2 / 5 / 2 | criterion and denoising loss weights |
| Denoising groups / box noise / label noise | 5 / 0.4 / 0.2 | `main.py`, `denoising.py` |
| Train short-side sizes | 480:32:800 | `datasets/coco.py` |
| Crop branch | Resize to 400/500/600, crop 384–600, then multi-scale resize | `datasets/coco.py` |
| Validation/test resize | Short side 800, long side ≤1333 | `datasets/coco.py` |
| Normalization | ImageNet mean/std from §4.2.1 | `datasets/coco.py` |

## Corrections relative to the supplied default source

1. **Classification coefficient:** the selected `main.py` default was 1. The release sets it to the manuscript's 2, and applies the same coefficient to denoising classification losses.
2. **Position temperature:** the selected entrypoint exposed temperature 20, but the backbone's positional-encoding builder ignored these arguments and constructed temperature 10000. The release actually passes 20 to the height and width sine encoding. This changes the numerical model behavior.
3. **Output classes:** the selected detector builder hard-coded 91 COCO outputs. The release uses an explicit class count (default 2), with reversible category-ID mapping. Single-class datasets can use 1. This changes head shapes and is incompatible with old 91-class checkpoints.
4. **GIoU geometry:** the selected utility clipped prediction corners into [0,1] and repaired box ordering before GIoU. The release computes GIoU on the actual corner coordinates, including corners outside the normalized image, consistent with the manuscript's GIoU definition. Existing small numerical denominator stabilizers remain. The change affects both matching and box losses.
5. **Portability:** denoising tensors follow the model device, replacing hard-coded CUDA allocations and a CPU scalar detour. The existing PyTorch reference MSDA implementation enables CPU checks. CUDA still uses the compiled extension and requires its installation.
6. **Release interfaces:** portable train/eval/inference commands replace machine-specific paths, duplicate experimental scripts, profiling side effects, and interactive debugger branches. No model weights are downloaded or bundled.

These source-preparation changes do not establish literal agreement with the manuscript. The deeper audit below identified remaining differences in positional encoding, area-based metrics, loss formulas, and parameter count. In particular, changing the output head from 91 to 2 is a model change; the manuscript does not itself specify the exact output-channel count. The original project and its experiment outputs are preserved outside this release.

## Validation boundary

The full model passed CPU synthetic-data checks for forward/backward execution, focal/L1/GIoU loss, denoising, empty targets, non-contiguous COCO category IDs, validation transforms and standard bounding-box evaluation. The tests use small model inputs for computation; the evaluation resize itself is checked separately. GPU extension compilation, GPU numerical equivalence, full-resolution training, original data partitions, and the paper's final metrics remain unverified by these checks.

The repository supplies the full proposed model and its training strategy. Baseline comparison implementations, all ablation variants, image-level screening analysis, and zero-shot label-harmonization protocols are outside this core release. Published numerical claims remain attributable to the original experiments.

## Module packaging

The paper modules are implemented as `MSFCA`, `HFF`, and `MSDA`, exposed from `models.nodule_detr`. The full detector is `NoduleDETR`, constructed by `build_nodule_detr`. HFF owns the feature projections and SCDown fusion operations, rather than leaving them inline in the detector.

Moving HFF into its own module changes its state-dictionary names: `input_proj.*` and `scDown1/2/3.*` are now under the `hff.` prefix. Pair training and inference with checkpoints from the same module layout. Existing flattened-layout checkpoints require an explicit key conversion; optimizer-resume compatibility across that layout change is not asserted.

The packaging was checked against a pre-refactor model using a fixed random seed and input. Parameter/buffer tensors, initial values, output logits and predicted boxes must match exactly after applying the HFF key-prefix mapping. The full CPU smoke check is also rerun after the refactor.

## Unresolved findings from the deeper audit

These findings supersede any earlier implication that passing a smoke check or passing a value of 20 established full manuscript agreement. Runtime model code was not changed during this audit.

1. **Size-specific AP:** §4.2.2 defines medium as 32² ≤ area < 64² and large as area ≥64². `datasets/coco_eval.py` constructs ordinary `COCOeval` without overriding area ranges, which uses the 96² boundary. APm and APl therefore use a different protocol. COCO's inclusive boundary conventions also differ at exact endpoints. Overall AP does not change solely from reassigning size ranges.
2. **Position formula:** equations (12)–(13) have denominator `T * 10000^(j/d_model)`, with a different odd-channel exponent in the cosine equation, and the text gives `d_model=64`. The current feature-map branch divides by `20^(2*floor(j/2)/128)`, concatenating 128 channels per axis into 256 channels. The active dynamic-anchor sine embedding separately uses base 10000 and produces 512 channels for four coordinates before projection. Passing 20 to the feature-map branch does not resolve these mathematical and dimensional differences. The unused two-stage proposal-encoding helper also contains 10000; it is distinct from the active decoder path.
3. **Parameter count:** `sum(p.numel() for p in model.parameters())` gives 50,093,797 for the default two-output head. Table 1 reports 50.25M. Keeping the same current architecture but using 91 outputs gives 50,253,730, which rounds to 50.25M. This supports a possible explanation involving the historical head size, but does not prove which model generated the paper results or how the paper counted parameters.
4. **Focal loss:** equation (9) omits an alpha factor. The code uses positive alpha 0.25 and negative alpha 0.75; at p=0.5 it yields losses of 0.0433217 and 0.1299651 rather than the equation's 0.1732868. Gamma=2 and the 2/5/2 outer coefficients agree. The alpha factor may have been omitted in the paper's abbreviated expression; removing it is a scientific change that needs provenance, not an automatic cleanup.
5. **GIoU stabilization:** the implementation adds a fixed 1e-6 to the union and enclosure denominators. Identical normalized 1-pixel boxes at an 800-pixel scale produce loss about 0.390233; the unmodified analytical GIoU definition gives zero. Identical 10-pixel boxes produce loss about 0.006359. This discrepancy was not caught by the earlier moderate-size-box smoke check.
6. **Details requiring disclosure:** HFF's two final transitions share `scDown2`, while `scDown3` is registered but unused (68,864 parameters). The backbone also uses SCDown on four dimension-changing shortcuts. The stem/stage-1 parameters, including three MSFCA blocks, are frozen despite local initialization; ordinary BatchNorm running statistics still update in training. The paper does not specify these details sufficiently to certify the exact historical configuration.
7. **Actual training objective:** six decoder outputs each contribute focal/L1/GIoU terms for matching and denoising, giving 36 weighted scalar terms. Equations (7)–(11) describe the loss components but do not spell out that full auxiliary objective. This is consistent with an inherited DN training design but should be documented explicitly.
8. **Coverage:** patient splits, mask-to-box preparation, complete ablations, zero-shot label harmonization, screening operating points, patient bootstrap, and GFLOPs/FPS reproduction are not established by this core repository. The current evaluation loader requires matching category IDs/names, so a two-class checkpoint cannot directly evaluate a single-class external dataset through this entrypoint.

Lower-impact findings: the cardinality diagnostic incorrectly treats the final real class as background, but is excluded from the weighted objective and is not a clinical screening metric; some docstrings still describe an extra no-object output channel or frozen BatchNorm even though the actual model uses neither. The inherited resize rounding can produce a 1334-pixel long side for a 407×100 input despite a nominal cap of 1333.

Confirmed independently: the grouped DCT agrees with an orthonormal 2D DCT reference; MSDA aggregates four constant levels to their expected mean and masks padded values correctly; image/box flip, resize and crop remain synchronized; StepLR decays both learning-rate groups by 0.1 at each 40-epoch boundary. The three main module data flows and default encoder/decoder attention dimensions agree with the method description. These checks do not validate the trained clinical or detection performance.

Before presenting the code as an exact paper reproduction, reconcile the final experimental model/head, positional formula and scope of temperature, and actual area-bin evaluation protocol against the original final experiment configuration. Do not silently alter them solely to make names or printed values resemble the manuscript.
