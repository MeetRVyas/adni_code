# models_cnn_transformer.py
import timm

RECOMMENDED_IMG_SIZES = {
    "tf_efficientnet_b4": 380, "tf_efficientnetv2_s": 300, "inception_v3": 299, "xception": 299,
    "vit_base_patch16_224": 224, "swin_base_patch4_window7_224": 224, "convnext_small": 224,
    "convnext_tiny": 224, "maxvit_tiny_224": 224, "resnet50": 224, "resnext50_32x4d": 224,
    "densenet121": 224, "coatnet_0_rw_224": 224, "resnet18": 224, "vgg16_bn": 224, "efficientnet_b0": 224,
    "mobilenetv3_large_100": 224, "vit_tiny_patch16_224": 224, "poolformer_s12": 224,
    "efficientformer_l1": 224,"mobilevit_s": 224,"ghostnet_100": 224
}


def get_img_size(model_name):
    name = model_name.split(".")[0]
    return RECOMMENDED_IMG_SIZES.get(name, 224)

def get_model(model_name, num_classes, pretrained=True):
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if "inception" in model_name and pretrained: model.aux_logits = False
        print(f"Loaded model: {model_name} | Pretrained: {pretrained} | Classes: {num_classes}")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}'.")
        raise e