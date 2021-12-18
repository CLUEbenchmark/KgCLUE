# 读取out里面的epoch文件绘制训练过程
# 通过选择实现: 每一个epoch画一张图或所有epoch都绘制在一张图内

import os
import shutil
import matplotlib
from decimal import Decimal
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def draw_one_pic(path, out_path):
    with open(path, 'r', encoding='utf-8') as f:
        steps = []
        loss_ = []
        acc_ = []
        while True:
            line = f.readline()
            if line:
                tmp = line.split(",")
                step = tmp[0].split(":")[1]
                loss = Decimal(tmp[1].split(":")[1]).quantize(Decimal('0.00'))
                acc = Decimal(float(tmp[2].split(":")[1])*100.0).quantize(Decimal('0.00'))
                steps.append(step)
                loss_.append(loss)
                acc_.append(acc)
            else:
                break

    x_axis_data = steps
    y_axis_data = loss_
    y_axis_data_ = acc_
    # plt.figure(figsize=(19.2,10.8))
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y_axis_data, color='b', alpha=0.8, linewidth=1.5, label='loss')
    plt.plot(x_axis_data, y_axis_data_, color='r', alpha=0.8, linewidth=1.5, label='acc %')

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签

    plt.legend(loc="upper right")
    plt.xlabel('steps')

    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(3000)

    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)


    plt.savefig(out_path)  # 保存该图片


def main(divided=False):
    filenames = os.listdir(train_out_dir)
    filenames = sorted(filenames)

    if divided:
        for idx, filename in enumerate(filenames):
            file_path = os.path.join(train_out_dir, filename)
            out_path = os.path.join(pic_out, './training_epoch_' + str(idx) + '.png')
            draw_one_pic(file_path, out_path)
    else:
        # 思路合并所有的文件
        all_in_f = open("./tmp.txt", 'w', encoding='utf-8')
        for idx, filename in enumerate(filenames):
            if filename.startswith("epoch"):
                x = open(os.path.join(train_out_dir, filename), "r", encoding='utf-8')  # 打开列表中的文件,读取文件内容
                all_in_f.write(x.read())  # 写入新建的log文件中
                x.close()
            else:
                all_in_f.close()
                raise ValueError("the path is not pure！！！")
        all_in_f.close()
        draw_one_pic("./tmp.txt", os.path.join(pic_out, './training_out.png'))
        os.remove("./tmp.txt")


if __name__ == '__main__':
    train_out_dir = './train_out'
    pic_out = './pic_out'
    if not os.path.exists(pic_out):
        os.makedirs(pic_out)
    else:
        # 清空文件夹
        shutil.rmtree(pic_out)
        os.makedirs(pic_out)

    divided = False
    main(divided)
