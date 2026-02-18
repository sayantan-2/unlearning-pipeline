from pipeline.unlearning import UnlearnerConfig


# ============================================================
# ResNet50
# ============================================================

resnet50_target_modules = [
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.conv3",
    "layer4.1.conv1",
    "layer4.1.conv2",
    "layer4.1.conv3",
    "layer4.2.conv1",
    "layer4.2.conv2",
    "layer4.2.conv3",
    "fc",
]

resnet50_config = UnlearnerConfig(
    epochs=5,
    lr=0.00019770545752611148,
    rank=4,
    alpha=54,
    lambda_retain=3.6500479489591697,
    target_modules=resnet50_target_modules,
)


# ============================================================
# MobileNetV2
# ============================================================

mobilenetv2_target_modules = [
    "blocks.5.2.conv_pw",
    "blocks.5.2.conv_pwl",
    "blocks.6.0.conv_pw",
    "blocks.6.0.conv_pwl",
    "conv_head",
    "classifier",
]

mobilenetv2_config = UnlearnerConfig(
    epochs=5,
    lr=0.0004084053844586405,
    rank=7,
    alpha=38,
    lambda_retain=3.701343149230949,
    target_modules=mobilenetv2_target_modules,
)


# ============================================================
# VGG16
# ============================================================

vgg16_target_modules = [
    "features.24",
    "features.26",
    "features.28",
    "classifier.0",
    "classifier.3",
    "classifier.6",
]

vgg16_config = UnlearnerConfig(
    epochs=5,
    lr=0.0003104449791152426,
    rank=47,
    alpha=41,
    lambda_retain=4.192257782041986,
    target_modules=vgg16_target_modules,
)

# ============================================================
# Vitbase
# ============================================================

vit_base_target_modules = [
    "blocks.10.attn.qkv",
    "blocks.10.attn.proj",
    "blocks.10.mlp.fc1",
    "blocks.10.mlp.fc2",
    "blocks.11.attn.qkv",
    "blocks.11.attn.proj",
    "blocks.11.mlp.fc1",
    "blocks.11.mlp.fc2",
    "head",
]

vit_base_config = UnlearnerConfig(
    epochs=5,
    lr=6.84941045972412e-05,
    rank=37,
    alpha=7,
    lambda_retain=2.562392399161846,
    target_modules=vit_base_target_modules,
)
