'''
根据pytracking生成的uav_bike1_time.txt中的时间，批量得到算法的FPS
# 作者：Python小哥
# 链接：https://www.zhihu.com/question/583574611/answer/2893356861
# 来源：知乎
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''
import os
from os import listdir
from os.path import join

# 设置待计算平均值的txt文件夹路径
folder_path = r'D:\XYL\3.Object tracking\4.不常用的跟踪算法\MKDNet\pytracking\tracking_results'
# 设置保存平均值的结果txt文件夹路径
result_path = r'D:\XYL\3.Object tracking\4.不常用的跟踪算法\MKDNet\pytracking\tracking_results'


for root, dirs, files in os.walk(folder_path):
    for num, dir in enumerate(dirs):
        tracker_path = join(folder_path, dir)
        result_file = join(result_path, dir + '_FPS.txt')  # 设置结果文件名
        # 遍历文件夹，处理每个txt文件
        with open(result_file, 'w') as f:
            avg_overall = []
            for i, file_name in enumerate(sorted(os.listdir(tracker_path))):
                if not file_name.endswith('_time.txt'):
                    continue

                # 读取txt文件
                with open(os.path.join(tracker_path, file_name), 'r') as data_file:
                    data = [float(line.strip()) for line in data_file.readlines()]

                # 计算平均值
                avg = sum(data) / len(data)  # 每个序列平均时间
                # avg = sum(data[1:]) / len(data[1:])  # 每个序列平均时间，排除每个序列的第一帧处理时间。但是差别不大
                seq_name = file_name.split('.')[0]
                # 输出结果到文件
                f.write(f'{i+1}\t{seq_name}\t{1/avg:.2f}\n')  # 这里i是表示第几个，算上了没有_time后缀的.txt, fps = 1/avg
        # 在上述代码中，通过os.listdir遍历文件夹，再根据文件名是否以_time.txt结尾来判断是否是需要处理的文件。
        # 对于每个txt文件，使用with open打开文件，读取数据后计算平均值，再将文件序号和平均值输出到结果文件中。
        # 如果需要绘制曲线图，可以使用matplotlib库。在上述代码中添加以下代码来实现：
        import matplotlib.pyplot as plt
        # 读取结果文件，提取文件序号和平均值
        with open(result_file, 'r') as f:
            lines = f.readlines()
            x = [int(line.split('\t')[0]) / 2 for line in lines]  # 读取结果文件的第一列, /2
            y = [float(line.split('\t')[2]) for line in lines]  # 读取结果文件的第三列
            fps = sum(y)/len(y)
        # 绘制曲线图
        plt.figure(num)
        plt.rcParams['xtick.direction'] = 'in'  # 设置xtick和ytick的方向：in、out、inout
        plt.rcParams['ytick.direction'] = 'in'
        plt.plot(x, y)
        plt.xlabel('Sequence')
        plt.ylabel('FPS')
        plt.axhline(y=fps, c='r', linestyle='dashed')  # 加一条平均速度参考线，平行于x轴
        plt.text(0, max(y)-4, 'Average FPS: {:.1f}'.format(fps), c='r')
        plt.savefig(result_path+'\\{}_FPS.jpg'.format(dir), dpi=300)
        plt.show()
        print('{} FPS is: {}'.format(dir, fps))
print('Figure saved at {}'.format(result_path))


# '''
# 计算单个算法的FPS
# '''
# # 设置待计算平均值的txt文件夹路径
# folder_path = r'D:\XYL\3.Object tracking\4.不常用的跟踪算法\MKDNet\pytracking\tracking_results\MKDNet-GOT10k'
# # 设置保存平均值的结果txt文件夹路径
# result_path = r'D:\XYL\3.Object tracking\4.不常用的跟踪算法\MKDNet\pytracking\tracking_results'
# tracker = folder_path.split('\\')[-1] # 计算平均值的算法名字
# result_file = join(result_path, tracker + '_result.txt')
# # 遍历文件夹，处理每个txt文件
# with open(result_file, 'w') as f:
#     avg_overall = []
#     for i, file_name in enumerate(sorted(os.listdir(folder_path))):
#         if not file_name.endswith('_time.txt'):
#             continue
#
#         # 读取txt文件
#         with open(os.path.join(folder_path, file_name), 'r') as data_file:
#             data = [float(line.strip()) for line in data_file.readlines()]
#
#         # 计算平均值
#         avg = sum(data) / len(data)  # 每个序列平均时间
#         seq_name = file_name.split('.')[0]
#         # 输出结果到文件
#         f.write(f'{i+1}\t{seq_name}\t{1/avg:.2f}\n')  # 这里i是表示第几个，算上了没有_time后缀的.txt, fps = 1/avg
# # 在上述代码中，通过os.listdir遍历文件夹，再根据文件名是否以_time.txt结尾来判断是否是需要处理的文件。
# # 对于每个txt文件，使用with open打开文件，读取数据后计算平均值，再将文件序号和平均值输出到结果文件中。
# # 如果需要绘制曲线图，可以使用matplotlib库。在上述代码中添加以下代码来实现：
# import matplotlib.pyplot as plt
# # 读取结果文件，提取文件序号和平均值
# with open(result_file, 'r') as f:
#     lines = f.readlines()
#     x = [int(line.split('\t')[0]) / 2 for line in lines]  # 读取结果文件的第一列, /2
#     y = [float(line.split('\t')[2]) for line in lines]  # 读取结果文件的第三列
#     fps = sum(y)/len(y)
# # 绘制曲线图
# # 设置xtick和ytick的方向：in、out、inout
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.plot(x, y)
# plt.xlabel('Sequence')
# plt.ylabel('FPS')
# plt.axhline(y=fps, c='r', linestyle='dashed')  # 加一条横线，平行于x轴
# plt.text(0, max(y)-4, 'Average FPS: {:.1f}'.format(fps), c='r')
# plt.show()
# print('{} FPS is: {}'.format(tracker, fps))
