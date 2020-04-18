import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import shutil
import random


# 自适应各种分辨率
# 本程序应用逻辑判断方法寻找关键点，稳定性非常高:测试方法：在temp文件夹里放截图screenshot.png并运行程序，看output.png的标记
# 采用查表法得到不同距离对应的比例系数，有能力计算误差反馈并自动修正误差~
# 主要利用cv2匹配棋子位置和棋子落点，准确率极高；_find_checker_loc(img_rgb(画面截图), img_checker(棋子图片),tan(方块边斜率=0.577))
# 利用最近两个平台中心连线中点在屏幕的位置固定的特点，可精确确定当前平台中心与上一个平台中心,以及算出上一次的跳跃误差
# _get_target_loc(img_rgb(画面截图), checker_loc(棋子落点位置), checker_LT_loc(棋子左上角位置), cen_loc(对称中心), tan, cos, sin, search_begin_row=400)

# 关于透视角度的参数，不要修改！
A = 577  # 某一方块宽
B = 1000  # 某一方块长
TAN = A / B
L = (A * A + B * B) ** 0.5
COS = B / L
SIN = A / L

# 1280x720分辨率下的“中心位置”,会自动按实际分辨率比例修正不要改动。
CEN_LOC = [374.75, 652.75]

# 截图保存名称
screenshot_filename = "screenshot"

work_path = os.getcwd()  # 获取当前目录
ORDER_START = work_path + '/file/adb.exe '  # 获取adb工具位置


def _read_screenshot(imgfile_name):
    """
    读取图像并裁剪、缩放成1280x720像素大小图片

    比如，一个3000x2000像素大小的图片
    ---------------------
    |                   |
    |                   |
    |                   |
    |                   |
    |         .         | x3000
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    ---------------------
            x2000

    首先把它裁剪成1280:720比例的图片：
        因为2000/720 > 3000/1280,也就是要对宽进行裁剪，高不变，
        那么就是宽应该变为 (3000/1280)*720 = 1687.5
        裁剪的时候以中心为基准
    ------------------
    |                |
    |                |
    |                |
    |                |
    |        .       | x3000
    |                |
    |                |
    |                |
    |                |
    |                |
    ------------------
          x1687.5
    然后再缩放为1280x720大小的图片即可.

    """
    target_pxs = (1280, 720)

    img_rgb = cv2.imread('temp\%s.png' % imgfile_name)
    ratios = [img_rgb.shape[0]/target_pxs[0], img_rgb.shape[1]/target_pxs[1]]
    i = 0 if (ratios[0] > ratios[1]) else 1
    r = ratios[1-i]
    seq = my_int((img_rgb.shape[i] - r*target_pxs[i])/2)
    fig_size = [[0, img_rgb.shape[0]], [0, img_rgb.shape[1]]]
    fig_size[i][0] += seq
    fig_size[i][1] -= seq
    img_rgb = img_rgb[fig_size[0][0]:fig_size[0]
                      [1], fig_size[1][0]:fig_size[1][1]]

    img = cv2.resize(
        img_rgb, (target_pxs[1], target_pxs[0]), interpolation=cv2.INTER_AREA)

    def point_trans(p):
        return (my_int(p[0]*r), my_int(p[1]*r))

    return img, point_trans

# 除非特殊情况，所有参数根据分辨率自适应，不需要修改任何参数


def _main():

    # 创建运行时所需文件夹
    for dir_name in ("temp", "K_array"):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    # 检查必需文件夹
    if not os.path.exists("file"):
        print("缺少含有必要文件的file文件夹")
        input("再见!回车键结束")
        exit()

    # 记录日志
    _init_log()
    _log("开始准备...", "EVENT")

    # 截图
    _get_screenshot(screenshot_filename)  # 截图并保存在temp文件夹
    img_rgb, _ = _read_screenshot(screenshot_filename)  # 读取截图文件

    k_array = K_array(img_rgb, TAN, COS, SIN)
    _log("准备完成！开始运行\n", "EVENT")

    img_checker = cv2.imread(r'file\checker1.jpg')
    img_gameover = cv2.imread(r'file\gameover.jpg')
    img_record = cv2.imread(r'file\record.jpg')

    _run(k_array, screenshot_filename, CEN_LOC,
         img_checker, img_gameover, img_record)


def my_int(num):
    return int(round(num))


class K_array:
    def __init__(self, img_rgb, tan, cos, sin, k_array_sep=5, fix_step=1, jump_k0_fn="初始参数.txt", jump_k_array_filename="jump_k_array.txt"
                 ):
        """
         img_rgb：cv2图片对象
         tan：参数
         cos：参数
         sin：参数
         jump_k0：初始跳跃参数
         k_array_sep：参数
         fix_step：
         k_nparray_filename=r"np_array_data.txt"

        """

        self.init_files(jump_k0_fn, jump_k_array_filename)  # 初始化工作文件夹

        self.jump_k0 = self._get_jump_k0()

        self.fix_step = fix_step

        self.k_array_sep = k_array_sep

        self.jump_k_array = self._get_jump_k_array(img_rgb, tan, cos, sin)
        self._prepare_plot()

    def init_files(self, jump_k0_fn, jump_k_array_filename):  # 检查工作环境和文件存在性，如果不存在则创建
        self.work_path = os.path.join(os.path.dirname(__file__), "K_array")

        self.work_dirs = {}
        for dir_name in ['fig', 'data']:
            self.work_dirs[dir_name] = os.path.join(self.work_path, dir_name)
            if not os.path.exists(self.work_dirs[dir_name]):
                os.mkdir(self.work_dirs[dir_name])

        self.jump_k0_fn = os.path.join(self.work_path, jump_k0_fn)
        self.jump_k_array_filename = os.path.join(
            self.work_dirs["data"], jump_k_array_filename)

    def _get_jump_k0(self):

        if not os.path.exists(self.jump_k0_fn):
            with open(self.jump_k0_fn, "w") as f:
                f.write("2.30\n\n初始参数越大，整体上蓄力越久")

        with open(self.jump_k0_fn, "r") as f:
            k0 = float(f.readline())

        return k0

    def _save_jump_k0(self):
        with open(self.jump_k0_fn, "w") as f:
            f.write("%f\n\n初始参数越大，整体上蓄力越久" % self.jump_k0)

    def _prepare_plot(self):
        def find_between(args, con):
            left_index = con.find(args[0])
            right_index = con.rfind(args[1])
            if (left_index + 1) * (right_index + 1):
                return con[left_index + len(args[0]):right_index]
        # 准备画k_array图
        self.max_num = -1
        fig_list = os.listdir(self.work_dirs["fig"])
        for f in fig_list:
            num_t = int(find_between(["_", "."], f))
            if num_t > self.max_num:
                self.max_num = num_t

        self.x_d = range(
            0, self.jump_k_array[0].shape[0] * self.k_array_sep, self.k_array_sep)
        plt.figure(figsize=(20, 3))

    def _get_jump_k_array(self, img, tan, cos, sin, ):
        # 准备跳跃参数向量
        jump_k_array = None

        jump_k_array_path = os.path.join(
            self.work_dirs["data"], self.jump_k_array_filename)

        if os.path.exists(jump_k_array_path) and os.path.exists(self.jump_k_array_filename+"k0"):

            np_k0 = float(
                open(self.jump_k_array_filename+"k0", "r").readline())
            if np_k0 != self.jump_k0:
                print("k0被修改了，重新生成np_array_data")
                jump_k_array = None
            else:
                jump_k_array = np.loadtxt(self.jump_k_array_filename)

        if jump_k_array is None:
            with open(self.jump_k_array_filename+"k0", "w") as f:
                f.write(str(self.jump_k0))

            fake_p = (img.shape[1] / self.k_array_sep,
                      img.shape[1] * tan / self.k_array_sep)
            k_num = my_int(_cal_dis((0, 0), fake_p, cos, sin))

            jump_k_array = np.array(
                [[-1] * k_num, [-2] * k_num, [self.jump_k0] * k_num])  # 极小值，极大值，当前值

            l_a = (110, 400)

            mid_range = (my_int(l_a[0]/self.k_array_sep),
                         my_int(l_a[1]/self.k_array_sep))
            for j in range(mid_range[0], mid_range[1]):
                jump_k_array[2][j] += (348 - j * self.k_array_sep) / 583
            for j in range(0, mid_range[0]):
                jump_k_array[2][j] = jump_k_array[2][mid_range[0]]
            for j in range(mid_range[1], jump_k_array[2].shape[0]):
                jump_k_array[2][j] = jump_k_array[2][mid_range[1]-1]

        return jump_k_array

    def _save_jump_k_array(self):
        self._save_jump_k0()
        np.savetxt(self.jump_k_array_filename, self.jump_k_array)
        with open(self.jump_k_array_filename+"k0", 'w') as f:
            f.write(str(self.jump_k0))
        shutil.copyfile(self.jump_k_array_filename,
                        self.jump_k_array_filename + "BAK")

    def _get_k(self, distance):
        i = my_int(distance / self.k_array_sep)
        return self.jump_k_array[2][i]

    def _fix_k(self, distance, err_dis):
        dk = 0
        i = my_int(distance / self.k_array_sep)
        self.pre_k = self.jump_k_array[2][i]
        if distance < 10:  # 原地跳了一下，没用
            return dk
        if err_dis >= 0:
            self.jump_k_array[0][i] = self.jump_k_array[2][i]
        else:
            self.jump_k_array[1][i] = self.jump_k_array[2][i]

        dk1 = err_dis * self.fix_step / distance
        dk2 = (self.jump_k_array[0][i] + self.jump_k_array[1]
               [i]) / 2 - self.jump_k_array[2][i]

        if abs(err_dis) > 10 or self.jump_k_array[0][i] > self.jump_k_array[1][i] or self.jump_k_array[0][i] < 0:
            dk = dk1
        else:
            dk = min(dk1, dk2)
        self.jump_k_array[2][i] += dk

        self.jump_k0 = self.jump_k_array[2][110:400].mean()
        self._save_jump_k_array()
        return dk

    def _plot_and_save(self):
        plt.axis(
            [0, self.x_d[-1], max(np.min(self.jump_k_array[2])-0.1, 0),
             min(np.max(self.jump_k_array[2])+0.1, 3 * self.jump_k_array[2][my_int(348/self.k_array_sep)])])
        plt.plot(self.x_d, self.jump_k_array[0], "r.")
        plt.plot(self.x_d, self.jump_k_array[1], "b.")
        plt.plot(self.x_d, self.jump_k_array[2], "g.")
        self.max_num = 1  # 如果要分别保存改成+1
        plt.savefig(os.path.join(
            self.work_dirs["fig"], "fig_%d.png") % (self.max_num))
        plt.clf()


def _run(k_array: K_array, imgfile_name, cen_loc, img_checker, img_gameover, img_record, max_row=170):

    # 定义一些工具函数

    def _jump(distance, k_array):
        # 根据跳跃距离及其对应的参数向量确定按压时间进行跳跃
        k = k_array._get_k(distance)
        press_time = my_int(distance * k)

        tap_point = (400+random.randint(-100, 100), 1000+random.randint(-100, 100),
                     400+random.randint(-100, 100), 1000+random.randint(-100, 100))
        cmd_str = ORDER_START + \
            'shell input swipe %d %d %d %d ' % tap_point + str(press_time)
        _cmd(cmd_str)  # 执行跳跃命令

        return tap_point

    def _check_gameover(img_rgb, img_gameover, last_output_rgb, point_trans_fun):

        # 如果在游戏截图中匹配到带"再玩一局"字样的模板，则循环中止
        res_gameover = cv2.matchTemplate(
            img_rgb, img_gameover, cv2.TM_CCOEFF_NORMED)
        min_valg, max_valg, min_locg, gameover_LT_loc = cv2.minMaxLoc(
            res_gameover)
        gameover_loc = (400, 1050)
        flag = 0
        if (max_valg > 0.70):
            _log("探测到重新开始按钮，本次游戏结束！", "EVENT")
            flag = 1
            gameover_loc = (my_int(gameover_LT_loc[0] + img_gameover.shape[1] / 2),
                            my_int(gameover_LT_loc[1] + img_gameover.shape[0] / 2))
            if last_output_rgb is None:
                last_output_rgb = img_rgb
        tap_point = (gameover_loc[0] + random.randint(-30, 30),
                     gameover_loc[1] + random.randint(-30, 30))
        new_tap_point = point_trans_fun(tap_point)
        cmd = ORDER_START + 'shell input tap %d %d' % new_tap_point
        img_rgb = cv2.circle(img_rgb, new_tap_point, 30,
                             (200, 200, 200), -1)  # 点击位置

        cv2.imwrite('temp\gameover%d_lastsence.png' % over_n, last_output_rgb)
        cv2.imwrite('temp\gameover%d.png' % over_n, img_rgb)
        _cmd(cmd)
        time.sleep(3)
        return flag

    # 循环直到游戏失败结束

    distance = 0
    i = 1  # 循环次数
    is_on = True  # 循环开关
    try_n = 0  # 出错重试次数
    over_n = 0  # 游戏结束次数

    last_output_rgb = None  # 预留上上次的截图位置
    while is_on:
        k_array._plot_and_save()

        _log("====================\n[第%d次操作]:" % i, "EVENT")

        _get_screenshot(imgfile_name)  # 下载手机截图

        img_rgb, point_trans = _read_screenshot(imgfile_name)  # 读取、裁剪手机截图
        checker_loc, checker_LT_loc = _find_checker_loc(
            img_rgb, img_checker, TAN)  # 寻找棋子位置

        if checker_loc is not None:
            try_n = 0
        else:  # 如果没找打小棋子

            # 可能游戏结束
            if _check_gameover(img_rgb, img_gameover, last_output_rgb, point_trans):
                i = 1
                over_n += 1
                continue
            else:
                try_n += 1
                cv2.imwrite('temp/no_checker%d.png' % try_n, img_rgb)
                img_checker = cv2.imread('file/checker%d.jpg' % (try_n % 3))

                if try_n >= 6:
                    _log("探测到小跳棋失败！", "ERROR")
                    is_on = False
                continue

        # ----------
        # 如果匹配到特殊方块们，修改时间
        res_recoder = cv2.matchTemplate(
            img_rgb, img_record, cv2.TM_CCOEFF_NORMED)
        if (cv2.minMaxLoc(res_recoder)[1] > 0.90):
            _log("发现唱片的存在！", "EVENT")
            sleep_duration = 2.5
        else:
            sleep_duration = random.uniform(1.4, 2.4)
        # ----------

        top_y, target_loc, pre_target_loc, err_dis = _get_target_loc(
            img_rgb, checker_loc, checker_LT_loc, cen_loc, TAN, COS, SIN)

        # 修正相应的k

        dk = k_array._fix_k(distance, err_dis)

        pre_dis = distance
        # 计算下一次的距离并跳跃
        distance = _cal_dis(checker_loc, target_loc, COS, SIN)

        print(k_array._get_k(distance))
        log_con = "上一次的目标位置为:[%d,%d]" % (pre_target_loc[0], pre_target_loc[1]) + "\n" + \
                  "上一次的跳跃距离为:%f" % pre_dis + "\n" + \
                  "棋子与上次偏差为:%f" % err_dis + "\n" + \
                  "修正数为:%f" % dk + "\n" + \
                  "修正后的k:%f" % k_array._get_k(pre_dis) + "\n" + \
                  "修正后的k0:%f" % k_array.jump_k0 + "\n\n" + \
                  "棋子位置为:[%d,%d]" % checker_loc + "\n" + \
                  "目标位置为:[%d,%d]" % (target_loc[0], target_loc[1]) + "\n" + \
                  "跳跃距离为:%f" % distance + "\n" + \
                  "转换系数k为:%f" % k_array._get_k(distance) + "\n"
        _log(log_con, "EVENT")
        tap_point = _jump(distance, k_array)
        _log("跳完！\n====================\n", "EVENT")
        last_output_rgb = _save_outputimg(
            img_rgb, top_y, target_loc, pre_target_loc, checker_loc, distance, tap_point)

        time.sleep(sleep_duration)
        i += 1

        if i >= max_row and max_row != -1:
            t = input("接近得分安全界限，分数过高可能不显示排名。是否继续? 输入\"Y\"继续50次,\"N\"停止,\"I\"永远继续")
            while True:
                if t in ("Y", "y"):
                    max_row += 50
                    print("继续玩50步")
                    is_on = True
                    break
                elif t in ("N", "n"):
                    is_on = False
                    break
                elif t in ("I", "i"):
                    max_row = -1
                    print("警告，不限制可能有封号风险")
                    time.sleep(3)
                    is_on = True
                    break
                else:
                    print("请输入Y或N或I:", end=' ')

    print("请随便玩几下故意失败结束！")
    input("任意键退出~")


def _init_log():  # 在log中加标记以便与历史纪录区分
    if not os.path.exists("log"):
        os.makedirs("log")
    log_con = "\n\n\n******* 初始化 *******\n"
    with open("log/Logs.log", "a") as f:
        f.write(log_con)


def _log(log_con, type_name, is_print=True):
    if is_print:
        print(log_con)

    with open("log/Logs.log", "a", encoding="utf-8") as f:
        f.write(log_con)

    with open("log/%s.log" % type_name, "a", encoding="utf-8") as f:
        f.write(log_con)


def _get_screenshot(name):
    path = 'temp/%s.png' % name
    if os.path.exists(path):
        shutil.copy(path, 'temp/last_%s.png' % name)
        os.remove(path)

    while True:
        print("开始截屏...", end="")
        _cmd(ORDER_START + 'shell screencap -p /sdcard/%s.png' % str(name))
        _cmd(ORDER_START + 'pull /sdcard/%s.png ./temp' % str(name))
        if os.path.exists(path):
            print("完成!")
            return True
        else:
            input("获取截图失败，请检查手机是否连接到电脑，并是否开启开发者模式,回车继续")


def _cmd(cmd_str):
    p = os.popen(cmd_str)
    p.read()
    # print(cmd_str,r.read(),sep="\n")


def _cal_dis(P1, P2, cos, sin):
    return _cal_dis_s(P1, P2, cos, sin)


def _cal_dis_n(P1, P2):
    return ((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2) ** 0.5


def _cal_dis_s(P1, P2, cos, sin):
    return ((P1[0] - P2[0]) ** 2 / (2 * (cos ** 2)) + (P1[1] - P2[1]) ** 2 / (2 * (sin ** 2))) ** 0.5


def _find_checker_loc(img_rgb, img_checker, tan):
    '''
    img_rgb:截图的cv2图像对象
    img_checker:棋子的cv2图像对象
    tan:表现透视角度的参数，计算棋子落脚点用
    '''
    hc, wc = img_checker.shape[0:2]
    res_checker = cv2.matchTemplate(img_rgb, img_checker, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, checker_LT_loc = cv2.minMaxLoc(res_checker)
    checker_loc = (my_int(
        checker_LT_loc[0] + wc / 2), my_int(checker_LT_loc[1] + hc - wc * tan / 2 + 2))

    if (max_val1 > 0.70):
        # 匹配度大于 70% 时认为匹配成功
        return checker_loc, checker_LT_loc
    else:
        _log("没有探测到小跳棋！匹配度为%f" % max_val1, "ERROR")
        return None, None


def _get_target_loc(img_rgb, checker_loc, checker_LT_loc, cen_loc, tan, cos, sin, c_sen=5, search_begin_row=400):
    top_y = None
    target_loc = None
    img_rgb.dtype = 'int8'
    # 避免把棋子顶端当作方块顶端
    if checker_loc[0] < img_rgb.shape[1] / 2:  # 如果棋子在屏幕左边，目标方块一定在棋子右边
        b = checker_LT_loc[0] + 51
        e = img_rgb.shape[1]
    else:  # 如果棋子在屏幕右边，目标方块一定在棋子左边
        b = 0
        e = checker_LT_loc[0]

    for i in range(search_begin_row, img_rgb.shape[0]):
        h = img_rgb[i][b:e]
        f_c = h[0]
        r = []
        for m in range(0, h.shape[0]):
            d = np.linalg.norm(f_c-h[m])
            if d > c_sen:  # 探测灵敏度
                r.append(m)

        if len(r):
            top_y = i
            x = np.mean(r) + b
            det_y = tan * abs(x - cen_loc[0]) - \
                abs(top_y - cen_loc[1])  # 利用绝对中心找到偏移量
            y = top_y + abs(det_y)
            target_loc = (my_int(x), my_int(y))
            break
    img_rgb.dtype = 'uint8'

    # 计算上一次的跳跃目标
    pre_target_loc = (
        my_int(2 * cen_loc[0] - target_loc[0]), my_int(2 * cen_loc[1] - target_loc[1]))

    # 检查上一次跳跃与完美的偏差
    err_dis = _cal_dis(pre_target_loc, checker_loc, cos, sin)
    if checker_loc[1] - pre_target_loc[1] < 0:
        err_dis = -err_dis

    return top_y, target_loc, pre_target_loc, err_dis


def _save_outputimg(img_rgb, top_y, target_loc, pre_target_loc, checker_loc, distance, tap_point, name="output"):
    # 将图片输出以供调试
    output_rgb = img_rgb
    output_rgb = cv2.circle(
        output_rgb, (target_loc[0], top_y), 10, (255, 255, 255), -1)  # 目标方块的顶点
    output_rgb = cv2.circle(output_rgb, target_loc, 10,
                            (0, 0, 255), -1)  # 目标方块的中心点
    output_rgb = cv2.circle(output_rgb, checker_loc,
                            10, (0, 255, 0), -1)  # 棋子的落脚点
    output_rgb = cv2.circle(output_rgb, pre_target_loc,
                            10, (255, 0, 0), -1)  # 当前方块的中心点

    output_rgb = cv2.line(
        output_rgb, tap_point[0:2], tap_point[2:4], (170, 170, 170), 30)  # 点击位置

    output_rgb = cv2.line(output_rgb, pre_target_loc,
                          target_loc, (0, 255, 255), 2)  # 目标方块中心点 与 当前方块中心点 连线

    def _cal_midpoint(p1, p2, d=(0, 0)):
        return (my_int((p1[0]+p2[0])/2+d[0]), my_int((p1[1]+p2[1])/2+d[1]))
    output_rgb = cv2.putText(output_rgb, "%.2f" % distance, _cal_midpoint(
        target_loc, pre_target_loc, (0, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imwrite(r'temp\%s.png' % name, output_rgb)
    return output_rgb


if __name__ == "__main__":
    _main()
