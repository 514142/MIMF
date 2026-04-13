import os
import sys
import numpy as np
from datetime import datetime

# 添加项目根目录到Python搜索路径
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from sesf_net import SESF_Fuse
from utils import (calculate_entropy,
                   calculate_avg_gradient,
                   calculate_psnr,
                   calculate_ssim,
                   save_image)


def image_fusion(img1, img2, pair_prefix, save_root=None,
                 attention="cse",
                 use_morphology=True,
                 use_guided_filter=True,
                 exp_name="full"):
    """
    单对图像融合
    """

    # 每次按当前实验配置实例化融合器
    sesf = SESF_Fuse(
        attention=attention,
        use_morphology=use_morphology,
        use_guided_filter=use_guided_filter
    )

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "fusion_results",
                                 datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 每个实验单独存放
    exp_root = os.path.join(save_root, exp_name)
    pair_save_dir = os.path.join(exp_root, pair_prefix)
    os.makedirs(pair_save_dir, exist_ok=True)

    save_image(img1, os.path.join(pair_save_dir, "input_1.png"), f"{pair_prefix}-输入图像1")
    save_image(img2, os.path.join(pair_save_dir, "input_2.png"), f"{pair_prefix}-输入图像2")

    fused, dm = sesf.fuse(img1, img2)

    dm_visual = (dm * 255).astype(np.uint8)
    save_image(dm_visual, os.path.join(pair_save_dir, "decision_map.png"), f"{pair_prefix}-决策图")
    save_image(fused, os.path.join(pair_save_dir, "fused_result.png"), f"{pair_prefix}-融合图")

    metrics = {
        "信息熵（EN）": calculate_entropy(fused),
        "平均梯度（AG）": calculate_avg_gradient(fused),
        "峰值信噪比（PSNR-输入1）": calculate_psnr(img1, fused),
        "峰值信噪比（PSNR-输入2）": calculate_psnr(img2, fused),
        "结构相似性（SSIM-输入1）": calculate_ssim(img1, fused),
        "结构相似性（SSIM-输入2）": calculate_ssim(img2, fused)
    }

    metric_path = os.path.join(pair_save_dir, "fusion_metrics.txt")
    with open(metric_path, "w", encoding="utf-8") as f:
        f.write(f"图像对：{pair_prefix} 融合指标结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"实验名称: {exp_name}\n")
        f.write(f"attention: {attention}\n")
        f.write(f"use_morphology: {use_morphology}\n")
        f.write(f"use_guided_filter: {use_guided_filter}\n")
        f.write("=" * 50 + "\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    print(f"✅ {pair_prefix} 处理完成，结果保存至：{pair_save_dir}\n")
    return fused, pair_save_dir


def batch_image_fusion(img_pairs, save_root=None,
                       attention="cse",
                       use_morphology=True,
                       use_guided_filter=True,
                       exp_name="full"):
    """
    批量融合
    """
    print(f"📦 开始批量处理，共 {len(img_pairs)} 对图像")
    print(f"🧪 当前实验配置: exp_name={exp_name}, attention={attention}, "
          f"use_morphology={use_morphology}, use_guided_filter={use_guided_filter}\n")

    for idx, (img1, img2, pair_prefix) in enumerate(img_pairs, 1):
        print(f"🔄 正在处理第 {idx}/{len(img_pairs)} 对：{pair_prefix}")
        try:
            image_fusion(
                img1=img1,
                img2=img2,
                pair_prefix=pair_prefix,
                save_root=save_root,
                attention=attention,
                use_morphology=use_morphology,
                use_guided_filter=use_guided_filter,
                exp_name=exp_name
            )
        except Exception as e:
            print(f"❌ {pair_prefix} 处理失败：{str(e)}\n")
            continue

    final_root = save_root if save_root else "fusion_results/当前时间文件夹"
    print(f"🎉 批量处理完成！结果根目录：{final_root}")