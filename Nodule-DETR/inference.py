import torch
from models.DABDETR import build_DABDETR
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "XX/config.json" # change the path of the model config
model_checkpoint_path = "XX/checkpoint199.pth" # change the path of the model checkpoint

args = SLConfig.fromfile(model_config_path)
model, criterion, postprocessors = build_dab_deformable_detr(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
id2name = {0: 'Aircraft',1: 'Ship',2: 'Car', 3: 'Bridge', 4: 'Tank', 5: 'Harbor'}
vslzr = COCOVisualizer()

from PIL import Image
import datasets.transforms as T
image = Image.open("XX/test2017/XX.jpg").convert("RGB") # Change the path of the image to be tested
# image
# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)

# predict images
output, _ = model(image[None],0)
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]
# visualize outputs
thershold = 0.3 # set a thershold

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir='XX')

