import torch
import torch.nn as nn


from transformers import Owlv2VisionConfig, Owlv2Processor, Owlv2VisionModel

class OWLV2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Owlv2VisionConfig.from_pretrained(self.vision_tower_name)
            

    def load_model(self):
        print("-------------load {} vision model----------------".format(self.vision_tower_name))
        self.image_processor = Owlv2Processor.from_pretrained(self.vision_tower_name).image_processor
        self.vision_tower = Owlv2VisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True



    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)


    @property
    def dtype(self):
        return self.vision_tower.dtype


    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

from torchvision import utils as vutils

if __name__ == "__main__":
    from PIL import Image
    import requests
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    # vision_tower = "/dev/shm/owlv2-base-patch16"
    # vision_tower = "/dev/shm/owlv2-large-patch14"
    vision_tower = "/dev/shm/owlv2-large-patch14-ensemble"

    model = OWLV2VisionTower(vision_tower, None)
    model.to(device)
    processor = model.image_processor

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # inputs = processor(images=image, return_tensors="pt")['pixel_values'][0].to(device, torch.float16)
    inputs = processor(images=image, return_tensors="pt")['pixel_values'][0]
    
    # vutils.save_image(inputs.clone().detach().to("cpu"), "test769.jpg")
    
    import pdb
    pdb.set_trace()
    a = 1

    qformer_outputs = model([inputs])

