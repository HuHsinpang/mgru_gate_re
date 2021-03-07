# 参考文献
# https://blog.csdn.net/htuhxf/article/details/82986440
# https://www.cnblogs.com/chenhuabin/tag/matplotlib/

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm._rebuild()
# font = fm.FontProperties(fname=r"C:\windows\fonts\STZHONGS.TTF")      # 华文中宋
font = fm.FontProperties(fname="/usr/share/fonts/win_fonts/STZHONGS.TTF")


# matplotlib plot()参数
science_colors = ['#0C5DA5', '#00B945', '#FF9500',
                  '#FF2C00', '#845B97', '#474747', '#9e9e9e']
ieee_colors = ['k', 'r', 'b', 'g']
bright_colors = ['#4477AA', '#EE6677', '#228833',
                 '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
high_contrast_colors = ['#004488', '#DDAA33', '#BB5566']
high_vis_colors = ["#0d49fb", "#e6091c",
                   "#26eb47", "#8936df", "#fec32d", "#25d7fd"]
light_colors = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB',
                '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD']
muted_colors = ['#CC6677', '#332288', '#DDCC77', '#117733',
                '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD']
retro_colors = ['#4165c0', '#e770a2',
                '#5ac3be', '#696969', '#f79a1e', '#ba7dcd']
vibrant_colors = ['#EE7733', '#0077BB', '#33BBEE',
                  '#EE3377', '#CC3311', '#009988', '#BBBBBB']
my_color = ieee_colors = ['r', 'c', 'm', 'g', 'b', 'k']

markers = ['o', 's', '^', 'v', '<', '>', 'd']
linestyles = ['-', '--', ':', '-.']

datasets = ['TENER', 'LRCNN', 'LGN', 'SoftLex', 'FLAT', 'AKE']


x_1 = [0.7,  0.725,0.73, 0.74, 0.75, 0.76, 0.77, 0.775,0.78, 0.79, 0.8,  0.81, 0.82, 0.825,0.83, 0.84, 0.85, 0.86, 0.87, 0.875,0.88, 0.89, 0.9,  0.91, 0.92, 0.925,0.93, 0.94, 0.95, 0.96, 0.97, 0.975,0.98,0.99, 1.0,  1.01, 1.02, 1.025,1.03, 1.04, 1.05, 1.06, 1.07, 1.075,1.08, 1.09, 1.1, 1.11, 1.12, 1.13]
y_1 = [84.02,84.21,83.97,84.23,83.70,83.93,84.61,84.38,84.48,83.88,85.04,84.12,83.69,84.24,83.81,84.09,84.29,84.38,84.37,84.11,84.37,84.52,84.31,84.11,84.09,84.28,84.23,84.66,84.39,84.46,84.24,84.25,84.6,83.99,84.51,84.64,84.32,84.3, 84.47,83.97,84.21,84.52,84.27,84.02,84.42,84.31,84.0,83.89,84.37,83.98]


# y_1 = [84.02,83.70,83.97,84.29,84.31,84.11,84.01,84.28,84.05,84.22,84.39,83.96,84.2,83.87,84.6,83.99,84.51,84.12,84.09,84.3, 84.47,83.97,84.21,83.96,84.04,84.02,84.07,84.31,84.0,83.89,84.07,83.98]
x_2 = [1,       2,      3,      4,      5,      6,      7]
y_2 = [84.32,   84.74,  84.71,  84.49,  84.56,  84.67,  84.23]


# 双图
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(
    8, 4), sharex=False, sharey=True, facecolor='white', dpi=100)
fig.suptitle('两种门控策略下不同β值对模型性能影响', fontproperties=font, fontsize=14)
ax1.plot(x_1, y_1, label='threshold strategy',
         ls=linestyles[0], marker=markers[0], color=science_colors[0])
ax1.set_xlabel('β', fontsize=12)
ax1.set_ylabel('F1(%)', fontsize=12)
ax1.legend()
ax1.autoscale(tight=True)

ax2.plot(x_2, y_2, label='truncation strategy',
         ls=linestyles[0], marker=markers[1], color=science_colors[1])
ax2.set_xlabel('β', fontsize=12)
ax2.legend()
ax2.autoscale(tight=True)

fig.subplots_adjust(wspace=0.05)
fig.savefig('./fig.pdf')
fig.savefig('./fig.jpg', dpi=700)
plt.show()
input()
