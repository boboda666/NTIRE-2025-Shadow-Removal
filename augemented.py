import os
import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt
from tqdm import tqdm


# ======================
# 配置参数
# ======================
class Config:
    shadow_dir = r"D:\CVPR\WSRD_DNSR_main\ntire2025_sh_rem_train\train\train_A"
    clean_dir = r"D:\CVPR\WSRD_DNSR_main\ntire2025_sh_rem_train\train\train_C"
    output_dir = r"D:\CVPR\WSRD_DNSR_main\nes"
    os.makedirs(output_dir, exist_ok=True)

    num_aug = 1
    crop_size = (1440, 1920)
    seed = 42


# ======================
# 同步增强管道定义
# ======================
def get_aug_pipeline():
    geo_pipeline = A.Compose([
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomResizedCrop(
            size=Config.crop_size,
            scale=(0.08, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        )
    ], additional_targets={'target': 'image'})

    color_pipeline = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.2),
            contrast_limit=(-0.2, 0.3),
            p=0.8
        ),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3)
    ])

    return geo_pipeline, color_pipeline


# ======================
# 动态阴影生成函数
# ======================
def add_dynamic_shadow(image):
    h, w = image.shape[:2]
    overlay = image.copy()

    num_vertices = np.random.randint(3, 8)
    vertices = np.array([[np.random.randint(0, w), np.random.randint(0, h)]
                         for _ in range(num_vertices)])

    alpha = np.random.uniform(0.3, 0.7)

    cv2.fillPoly(overlay, [vertices], (0, 0, 0))
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


# ======================
# 核心增强流程
# ======================
def process_pair(shadow_path, clean_path, save_prefix):
    shadow_img = cv2.cvtColor(cv2.imread(shadow_path), cv2.COLOR_BGR2RGB)
    clean_img = cv2.cvtColor(cv2.imread(clean_path), cv2.COLOR_BGR2RGB)

    geo_pipe, color_pipe = get_aug_pipeline()

    for aug_idx in range(Config.num_aug):
        if np.random.rand() < 0.5:
            shadow_img = add_dynamic_shadow(shadow_img.copy())

        geo_aug = geo_pipe(image=shadow_img, target=clean_img)
        aug_shadow = geo_aug["image"]
        aug_clean = geo_aug["target"]

        color_aug = color_pipe(image=aug_shadow)
        final_shadow = color_aug["image"]

        pair_id = f"{os.path.basename(shadow_path).split('.')[0]}_aug{aug_idx}"
        # 修改点1/2：改为PNG格式并优化压缩参数
        cv2.imwrite(
            os.path.join(Config.output_dir, f"{pair_id}_input.png"),
            cv2.cvtColor(final_shadow, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 无损压缩
        )
        cv2.imwrite(
            os.path.join(Config.output_dir, f"{pair_id}_target.png"),
            cv2.cvtColor(aug_clean, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )


# ======================
# 可视化验证函数（无需修改）
# ======================
def visualize_augmentation(shadow_path, clean_path):
    shadow_img = cv2.cvtColor(cv2.imread(shadow_path), cv2.COLOR_BGR2RGB)
    clean_img = cv2.cvtColor(cv2.imread(clean_path), cv2.COLOR_BGR2RGB)

    geo_pipe, color_pipe = get_aug_pipeline()

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for i in range(3):
        if i == 0:
            disp_shadow = shadow_img.copy()
            disp_clean = clean_img.copy()
        else:
            if np.random.rand() < 0.5:
                disp_shadow = add_dynamic_shadow(shadow_img.copy())
            geo_aug = geo_pipe(image=disp_shadow, target=clean_img)
            color_aug = color_pipe(image=geo_aug["image"])
            disp_shadow = color_aug["image"]
            disp_clean = geo_aug["target"]

        axes[i, 0].imshow(disp_shadow)
        axes[i, 0].set_title(f"Input (Aug #{i})")
        axes[i, 1].imshow(disp_clean)
        axes[i, 1].set_title(f"Target (Aug #{i})")

    plt.tight_layout()
    plt.show()


# ======================
# 主执行流程
# ======================
if __name__ == "__main__":
    np.random.seed(Config.seed)

    shadow_images = sorted([os.path.join(Config.shadow_dir, f)
                            for f in os.listdir(Config.shadow_dir)])
    clean_images = sorted([os.path.join(Config.clean_dir, f)
                           for f in os.listdir(Config.clean_dir)])

    assert len(shadow_images) == len(clean_images), "图像数量不匹配"

    print(f"开始增强 {len(shadow_images)} 对图像...")
    for shadow_path, clean_path in tqdm(zip(shadow_images, clean_images)):
        base_name = os.path.basename(shadow_path).split('.')[0]
        process_pair(shadow_path, clean_path, base_name)

    idx = np.random.randint(len(shadow_images))
    visualize_augmentation(shadow_images[idx], clean_images[idx])
    print(f"增强完成！结果保存在 {Config.output_dir}")