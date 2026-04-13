import os
import cv2
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import random

def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, learning_rate, epoch):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-5"""
    lr = learning_rate * (0.8 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_loss(experiment_name, epoch, train_loss_list, val_loss_list):
    clear_output(True)
    print('Epoch %s. train loss: %s. val loss: %s' % (epoch, train_loss_list[-1], val_loss_list[-1]))
    print('Best val loss: %s' % (min(val_loss_list)))
    print('Back up')
    print('train_loss_list: {}'.format(train_loss_list))
    print('val_loss_list: {}'.format(val_loss_list))
    plt.figure()
    plt.plot(train_loss_list, color="r", label="train loss")
    plt.plot(val_loss_list, color="b", label="val loss")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name, fontsize=16)
    figure_address = os.path.join(os.path.join(os.getcwd(), 'nets'), 'figures')
    os.makedirs(figure_address, exist_ok=True)  # 新增：确保文件夹存在
    plt.savefig(os.path.join(figure_address, experiment_name + '_loss.png'))
    plt.close()  # 新增：关闭图像，释放内存

def plot_iteration_loss(experiment_name, epoach, loss, lp_loss, lssim_loss):
    plt.figure()
    plt.plot(loss, color="r", label="loss")
    plt.plot(lp_loss, color="g", label="lp_loss")
    plt.plot(lssim_loss, color="b", label="lssim_loss")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name, fontsize=16)
    figure_address = os.path.join(os.path.join(os.getcwd(), 'nets'), 'figures')
    os.makedirs(figure_address, exist_ok=True)  # 新增：确保文件夹存在
    plt.savefig(os.path.join(figure_address, experiment_name + '_' + str(epoach) + '_loss.png'))
    plt.close()  # 新增：关闭图像，释放内存

def print_and_log(content, is_out_log_file=True, file_address=None):
    print(content)
    if is_out_log_file and file_address is not None:
        os.makedirs(os.path.dirname(file_address), exist_ok=True)  # 新增：确保日志文件夹存在
        with open(file_address, "a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")

def get_mean_value(input_dir):
    images_list = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    count = 0
    pixel_sum = 0
    for index, sub_folder in enumerate(images_list):
        if not os.path.isdir(sub_folder):  # 新增：跳过非文件夹（避免文件干扰）
            continue
        image_name = os.path.basename(sub_folder)
        img1_path = os.path.join(sub_folder, image_name + "_1.png")
        img2_path = os.path.join(sub_folder, image_name + "_2.png")
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):  # 新增：检查文件存在
            continue
        last_image = cv2.imread(img1_path, 0) * 1.0 / 255
        next_image = cv2.imread(img2_path, 0) * 1.0 / 255
        pixel_sum += np.sum(last_image) + np.sum(next_image)
        count += last_image.size + next_image.size
    return pixel_sum / count if count != 0 else 0.5  # 新增：避免除零错误

def get_std_value(input_dir, mean):
    images_list = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    count = 0
    pixel_sum = 0
    for index, sub_folder in enumerate(images_list):
        if not os.path.isdir(sub_folder):  # 新增：跳过非文件夹
            continue
        image_name = os.path.basename(sub_folder)
        img1_path = os.path.join(sub_folder, image_name + "_1.png")
        img2_path = os.path.join(sub_folder, image_name + "_2.png")
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):  # 新增：检查文件存在
            continue
        last_image = np.power((cv2.imread(img1_path, 0) * 1.0 / 255) - mean, 2)
        next_image = np.power((cv2.imread(img2_path, 0) * 1.0 / 255) - mean, 2)
        pixel_sum += np.sum(last_image) + np.sum(next_image)
        count += last_image.size + next_image.size
    return np.sqrt(pixel_sum / count) if count != 0 else 0.2  # 新增：避免除零错误

# 新增：通用图像保存函数（支持中文路径、过程图处理）
def save_image(img, save_path, img_desc="image"):
    """
    保存图像（兼容中文路径，支持灰度/彩色图）
    :param img: 图像数组（np.ndarray）
    :param save_path: 保存路径（含中文）
    :param img_desc: 图像描述（用于打印日志）
    """
    try:
        # 处理灰度图（单通道→3通道，便于统一保存）
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 兼容中文路径保存
        cv2.imencode(os.path.splitext(save_path)[1], img)[1].tofile(save_path)
        print(f"✅ {img_desc}已保存至：{save_path}")
    except Exception as e:
        print(f"❌ {img_desc}保存失败：{str(e)}")