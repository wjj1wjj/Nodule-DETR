# Nodule-DETR

Code for **Nodule-DETR: A Novel DETR Architecture with Frequency-Channel Attention for Thyroid Nodule Detection and Benign-Malignant Classification in Ultrasound Images**.

This source release contains the full model, its paper training configuration, COCO bounding-box evaluation, and image inference. Datasets, trained weights, experiment outputs, manuscript files, and historical model variants are excluded.

The implementation was checked against the accepted revised manuscript. See [paper-to-code correspondence](docs/paper_alignment.md) for the module mapping, source-code corrections, and validation limits. Reported paper scores have not been re-established by this release's smoke checks.

## Known correspondence gaps

A deeper manuscript audit found unresolved differences: the paper defines medium/large AP at a 64×64 boundary, while this evaluator uses COCO's 96×96 default; the position-encoding implementation does not implement equations (12)–(13) literally; and the default two-output model has 50.093797M parameters rather than Table 1's 50.25M. The alpha-weighted focal loss and stabilized GIoU also require reconciliation with the written formulas. See [the detailed correspondence notes](docs/paper_alignment.md#unresolved-findings-from-the-deeper-audit). The current source is not certified as an exact implementation of every manuscript formula or reported experiment.

## Paper modules

```text
models/nodule_detr/
├── model.py                    # NoduleDETR and build_nodule_detr
├── backbone.py                 # Multi-stage feature extraction
├── resnet_msfca.py              # ResNet50 with MSFCA residual blocks
├── msfca.py                     # MSFCA: frequency-channel attention
├── hff.py                       # HFF: projections, P4 and bottom-up SCDown fusion
├── msda.py                      # MSDA: multi-scale deformable attention
├── deformable_transformer.py    # Six-layer encoder and decoder
├── denoising.py                 # Training-only query denoising and its losses
└── ops/                        # Low-level deformable sampling kernels
```

The model follows `MSFCA backbone → HFF → MSDA encoder/decoder → prediction heads`.
The public Python interface exports `NoduleDETR`, `build_nodule_detr`, `MSFCA`, `HFF` and `MSDA` from `models`. Source copyright and upstream attribution headers are retained.

## Setup

Use Python 3.8 and install the dependencies in `requirements.txt`. The CPU smoke check used PyTorch 2.4.0 and torchvision 0.19.0. For GPU training, install the corresponding PyTorch CUDA build for your environment before installing the remaining dependencies. A matching CUDA toolkit and C++ compiler are required to build the deformable-attention extension.

```bash
python -m pip install -r requirements.txt
cd models/nodule_detr/ops
python setup.py build_ext --inplace
python -m pip install . --no-build-isolation
cd ../../..
```

The extension is used on CUDA tensors. CPU checks use the supplied PyTorch reference implementation of the same deformable sampling operation. The GPU extension has not been rebuilt or exercised as part of the CPU release checks.

No weights are downloaded automatically. The selected source backbone initializes its parameters locally; this release preserves that initialization behavior.

## Data

Prepare existing train/validation/test partitions in COCO detection format:

```text
data/coco/
├── train2017/
├── val2017/
├── test2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    └── instances_test2017.json
```

Each annotation needs `image_id`, `category_id`, and `bbox` in pixel `[x, y, width, height]` format, with a unique annotation `id`. Include `area` and `iscrowd` for evaluation. Images need unique IDs, `file_name`, `width`, and `height`. Keep identical category IDs and names in all splits, including categories absent from a particular split. Negative images have no lesion annotations; background is not an object category.

The loader sorts category IDs, maps them to contiguous model indices, and maps predictions back to the original IDs before evaluation. Category names and mappings are stored with each checkpoint. The default is two object classes for benign/malignant classification. Use `--num_classes 1` for a dataset annotated as a single nodule class; this trains a separate detection model and does not create benign/malignant labels that the dataset lacks.

Supply the partitions used in the paper: patient-level splits for the private dataset and the stated public dataset splits. This repository does not regenerate or certify those splits. Convert segmentation masks into lesion boxes before use; no medical images or annotation files are distributed here.

## Train

```bash
python main.py --coco_path data/coco --output_dir outputs/nodule_detr
```

Defaults follow the manuscript: batch size 2, 200 epochs, AdamW, learning rates `1e-4` / `1e-5` for non-backbone/backbone parameters, weight decay `1e-4`, and StepLR with a factor of 0.1 every 40 epochs. The loss weights are focal/L1/GIoU = 2/5/2. Query denoising uses 5 groups, box noise 0.4, and label noise 0.2.

Training uses the paper's horizontal flip, multi-scale resize, crop branch and ImageNet normalization. Validation/testing resize the short side to 800 with a maximum long side of 1333. See `datasets/coco.py` for the complete transform definition.

Training runs on one GPU in float32. The backbone stem and stage 1 retain the source implementation's parameter freezing; stage 2–4 and the detector are trainable. Backbone BatchNorm layers retain the source behavior. Seed 42, gradient clipping at 0.1, auxiliary decoder losses and FFN size 2048 are inherited implementation settings; the manuscript does not specify all of these details.

The output directory contains the configuration, category definitions, per-epoch log, last checkpoint, best validation-AP checkpoint, and periodic checkpoints. These files are generated only when the user trains; they are ignored by Git. Checkpoint selection by validation AP is an engineering convention in this release, not a newly established paper evaluation protocol.

To resume, provide the same scientific arguments as the original run:

```bash
python main.py --coco_path data/coco --output_dir outputs/nodule_detr \
  --resume outputs/nodule_detr/checkpoint_last.pth
```

## Evaluate and infer

```bash
python main.py --coco_path data/coco --output_dir outputs/test \
  --resume outputs/nodule_detr/checkpoint_best.pth --eval --eval_split test

python inference.py --checkpoint outputs/nodule_detr/checkpoint_best.pth \
  --image example.png --output outputs/prediction.json --threshold 0.5
```

Evaluation returns the 12 standard COCO bounding-box AP/AR statistics, including AP, AP50, AP75, APs, APm and APl. The model has 300 queries; the inherited postprocessor selects the top 100 query/class scores for COCO evaluation, without NMS. Image inference exports category names, confidence scores, and original-image pixel `box_xyxy` coordinates. Its threshold of 0.5 is a user-facing display filter, not a claim about the paper's screening operating point.

Checkpoints must be created by this release. The legacy 91-output head and raw category IDs are incompatible with the default two-output mapped head; do not load them with `strict=False` and treat that as a faithful conversion. Only load trusted PyTorch checkpoints.

The evaluation command assumes the checkpoint and evaluation set have the same label semantics. Cross-dataset zero-shot category harmonization and screening-specific sensitivity/specificity calculations are not supplied by this core release.

## Check the source release

```bash
python tools/smoke_test.py
python tools/check_release.py
```

The smoke check uses temporary synthetic images and the full default model on CPU. It exercises forward/backward computation, denoising, empty targets, category mapping, and COCO evaluation at small image sizes. It does not run a 200-epoch experiment or validate medical performance.

## Acknowledgements and license

The implementation derives from the existing DN-DETR / DAB-DETR project, Deformable DETR, DETR, and the upstream multi-spectral channel-attention implementation. Spatial-channel decoupled downsampling follows the existing SCDown helper in the supplied project. Original source attribution headers and the project's [LICENSE](LICENSE) are retained. Unused experimental attention, Mamba, deformable-convolution and visualization branches are excluded.
