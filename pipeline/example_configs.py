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
    epochs=3,
    lr=0.0004918987182238827,
    rank=8,
    alpha=18,
    lambda_retain=3.6500479489591697,
    lambda_hinge=0.16988108526106105,
    lambda_kl=4.264085465184694,
    hinge_margin=11.416722631389227,
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
    epochs=8,
    lr=0.000249097203632189,
    rank=44,
    alpha=30,
    lambda_retain=4.192257782041986,
    lambda_hinge=0.20852382593222335,
    lambda_kl=1.8264131111869117,
    hinge_margin=6.376586775265366,
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
