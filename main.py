import sys
import os
import argparse
from utils import batch_pair_images
from algo import batch_image_fusion


def parse_args():
    parser = argparse.ArgumentParser(description="多聚焦图像批量融合（支持消融实验）")

    # 必选参数
    parser.add_argument("--data_dir", required=True, help="数据集文件夹路径（含所有待配对图像）")

    # 可选参数
    parser.add_argument("--save_root", default=None, help="结果保存根目录（默认：fusion_results/当前时间）")
    parser.add_argument("--suffix1", default="_1", help="图像对1的后缀（默认：_1，如 imgA_1.png）")
    parser.add_argument("--suffix2", default="_2", help="图像对2的后缀（默认：_2，如 imgA_2.png）")

    # attention 类型
    parser.add_argument("--attention", default="cse", choices=["cse", "sse", "scse"],
                        help="注意力模块类型（默认：cse）")

    # 消融开关
    parser.add_argument("--no_morphology", action="store_true", help="关闭形态学细化")
    parser.add_argument("--no_guided_filter", action="store_true", help="关闭引导滤波")

    # 实验名
    parser.add_argument("--exp_name", default="full", help="实验名称，用于结果目录命名")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 检查数据集目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据集文件夹不存在：{args.data_dir}")
        sys.exit(1)

    # 2. 自动配对图像
    print(f"🔍 正在数据集 {args.data_dir} 中配对图像（后缀：{args.suffix1}/{args.suffix2}）...\n")
    try:
        img_pairs = batch_pair_images(
            data_dir=args.data_dir,
            suffix1=args.suffix1,
            suffix2=args.suffix2
        )
        if not img_pairs:
            print("❌ 未找到有效图像对，程序退出！")
            sys.exit(0)
    except Exception as e:
        print(f"❌ 图像配对失败：{str(e)}")
        sys.exit(1)

    # 3. 批量融合
    print("🚀 开始批量融合...\n")
    batch_image_fusion(
        img_pairs=img_pairs,
        save_root=args.save_root,
        attention=args.attention,
        use_morphology=not args.no_morphology,
        use_guided_filter=not args.no_guided_filter,
        exp_name=args.exp_name
    )


if __name__ == "__main__":
    main()