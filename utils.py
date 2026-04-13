import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim_metric


def image_read_from_chinese_path(image_file_name):
    """读取中文路径图像（兼容灰度/彩色）"""
    try:
        image_numpy_data = cv2.imdecode(np.fromfile(image_file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_numpy_data is None:
            raise ValueError(f"图像为空：{image_file_name}")
        return image_numpy_data
    except Exception as e:
        raise Exception(f"读取失败：{str(e)}")


# ------------------------------
# 4个核心评估指标（不变）
# ------------------------------
def calculate_entropy(image):
    """信息熵（EN）：衡量信息量"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))


def calculate_avg_gradient(image):
    """平均梯度（AG）：衡量细节清晰度"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2).mean()


def calculate_psnr(image1, image2):
    """峰值信噪比（PSNR）：衡量保真度"""
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(image1, image2):
    """结构相似性（SSIM）：衡量结构一致性"""
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return ssim_metric(image1, image2, data_range=255)


# ------------------------------
# 新增：批量图像自动配对（核心函数）
# ------------------------------
def batch_pair_images(data_dir, suffix1="_1", suffix2="_2", valid_exts=[".png", ".jpg", ".jpeg"]):
    """
    按“前缀+后缀”自动配对图像（如imgA_1.png ↔ imgA_2.png）
    :param data_dir: 数据集文件夹路径（含所有待配对图像）
    :param suffix1: 图像对1的后缀（默认"_1"）
    :param suffix2: 图像对2的后缀（默认"_2"）
    :param valid_exts: 有效图像格式
    :return: img_pairs: 配对列表，每个元素为（img1数组, img2数组, 前缀）
    """
    # 1. 遍历文件夹，收集所有有效图像路径
    img_paths = {}  # 键：前缀，值：{suffix1:路径, suffix2:路径}
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_name, ext = os.path.splitext(file)
            # 过滤无效格式
            if ext.lower() not in valid_exts:
                continue
            # 提取前缀（去除后缀）
            if file_name.endswith(suffix1):
                prefix = file_name[:-len(suffix1)]
                if prefix not in img_paths:
                    img_paths[prefix] = {}
                img_paths[prefix][suffix1] = os.path.join(root, file)
            elif file_name.endswith(suffix2):
                prefix = file_name[:-len(suffix2)]
                if prefix not in img_paths:
                    img_paths[prefix] = {}
                img_paths[prefix][suffix2] = os.path.join(root, file)

    # 2. 过滤无效配对（确保每对有且仅有2张图）
    img_pairs = []
    invalid_prefixes = []
    for prefix, paths in img_paths.items():
        if suffix1 in paths and suffix2 in paths:
            # 读取图像
            try:
                img1 = image_read_from_chinese_path(paths[suffix1])
                img2 = image_read_from_chinese_path(paths[suffix2])
                img_pairs.append((img1, img2, prefix))
            except Exception as e:
                invalid_prefixes.append(f"{prefix}（读取失败：{str(e)}）")
        else:
            invalid_prefixes.append(f"{prefix}（缺失{suffix1}或{suffix2}图像）")

    # 3. 打印配对结果日志
    print(f"📊 图像配对结果：")
    print(f"✅ 有效配对：{len(img_pairs)} 对")
    if invalid_prefixes:
        print(f"❌ 无效配对（共{len(invalid_prefixes)}个）：")
        for info in invalid_prefixes[:5]:  # 只打印前5个，避免日志过长
            print(f"  - {info}")
        if len(invalid_prefixes) > 5:
            print(f"  - ... 还有{len(invalid_prefixes) - 5}个无效配对未显示")
    print()
    return img_pairs


# ------------------------------
# 图像保存工具（不变，适配批量）
# ------------------------------
def save_image(img, save_path, img_desc="image"):
    try:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            save_path = os.path.splitext(save_path)[0] + '.png'
        success, buf = cv2.imencode(ext, img)
        if not success:
            raise ValueError(f"编码失败：{save_path}")
        buf.tofile(save_path)
        # 简化日志（批量处理时避免冗余）
    except Exception as e:
        raise Exception(f"{img_desc}保存失败：{str(e)}")