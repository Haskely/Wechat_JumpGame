import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import shutil

#除非特殊情况，所有参数自适应，不需要修改任何参数
def _main():
    # 关于透视角度的参数，不要修改！
    a = 577
    b = 1000
    tan = a / b
    L = (a * a + b * b) ** 0.5
    cos = b / L
    sin = a / L

    # 1280x720分辨率下的“中心位置”,会自动修正不要改！！
    cen_loc = [374.75, 652.75]

    # 截图保存名称
    imgfile_name = "screenshot"

    _init_log()
    _log("开始准备...","EVENT")
    _get_screenshot(imgfile_name)
    img_rgb = cv2.imread('temp\%s.png' % imgfile_name)
    last_output_rgb = img_rgb

    fix_kx = img_rgb.shape[0] / 1280
    fix_ky = img_rgb.shape[1] / 720
    cen_loc = _fix_cen_loc(cen_loc, fix_kx, fix_ky)
    _log("全局分辨率对应cen_loc转换后为[%f,%f]" % (cen_loc[0],cen_loc[1]), "EVENT")

    k_array = K_array(img_rgb, tan, cos, sin, fix_kx, fix_ky)
    _log("准备完成！开始运行\n", "EVENT")
    _run(k_array, imgfile_name, tan, cos, sin, last_output_rgb, cen_loc)


def _run(k_array, imgfile_name, tan, cos, sin, last_output_rgb, cen_loc):
    img_checker = cv2.imread(r'file\checker1.jpg')
    hc, wc = img_checker.shape[0:2]
    img_gameover = cv2.imread(r'file\gameover.jpg')
    img_record = cv2.imread(r'file\record.jpg')
    # 循环直到游戏失败结束
    max_row = 1000
    distance = 0
    i = 0
    is_on = True
    try_n = 0
    over_n = 0
    while is_on:
        k_array._plot_and_save()
        if i >= max_row:
            is_on = False
            break
        _log("第%d次" % i, "EVENT")

        _get_screenshot(imgfile_name)

        img_rgb = cv2.imread(r'temp\%s.png' % imgfile_name)

        # 如果在游戏截图中匹配到带"再玩一局"字样的模板，则循环中止
        res_gameover = cv2.matchTemplate(img_rgb, img_gameover, cv2.TM_CCOEFF_NORMED)
        if (cv2.minMaxLoc(res_gameover)[1] > 0.90):
            _log("游戏结束！", "EVENT")
            over_n += 1
            cv2.imwrite(r'temp\gameover%d.png' % over_n, last_output_rgb)

            cmd = 'adb shell input tap 372 1055'
            _cmd(cmd)
            i = 0
            time.sleep(3)

        # 模板匹配截图中小跳棋的位置
        res_checker = cv2.matchTemplate(img_rgb, img_checker, cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, cheater_LT_loc = cv2.minMaxLoc(res_checker)
        checker_loc = (my_int(cheater_LT_loc[0] + wc / 2), my_int(cheater_LT_loc[1] + hc - wc * tan / 2 + 2))

        if (cv2.minMaxLoc(res_checker)[1] > 0.70):
            try_n = 0
        else:
            try_n += 1
            _log("没有探测到小跳棋！匹配度为%f" % max_val1, "ERROR")
            cv2.imwrite(r'temp\no_checker.png', img_rgb)
            time.sleep(3)
            img_checker = cv2.imread(r'file\checker%d.jpg' % (try_n % 3))
            hc, wc = img_checker.shape[0:2]
            if try_n >= 6:
                _log("探测到小跳棋失败！", "ERROR")
                is_on = False
            continue

        # 如果匹配到唱片，增加延长时间
        res_recoder = cv2.matchTemplate(img_rgb, img_record, cv2.TM_CCOEFF_NORMED)
        if (cv2.minMaxLoc(res_recoder)[1] > 0.90):
            _log("发现唱片的存在！", "EVENT")
            sleep_duration = 1.5
        else:
            sleep_duration = 1.5

        top_y, target_loc, pre_target_loc, err_dis = _get_target_loc(img_rgb, checker_loc, cheater_LT_loc, cen_loc, tan,cos,sin)

        # 修正相应的k

        k_array._fix_k(distance, err_dis)

        p_dis = distance
        # 计算下一次的距离并跳跃
        distance = _cal_dis(checker_loc, target_loc, cos, sin)
        last_output_rgb = _save_outputimg(img_rgb, top_y, target_loc, pre_target_loc, checker_loc)

        log_con = "上一次的目标位置为:[%d,%d]" % (pre_target_loc[0], pre_target_loc[1]) + "\n" + \
                  "上一次的跳跃距离为:%f" % p_dis + "\n" + \
                  "棋子与上次偏差为:%f" % err_dis + "\n" + \
                  "修正数为:%f" % (k_array._get_k(p_dis) - k_array.pre_k) + "\n" + \
                  "修正后的k:%f" % k_array._get_k(p_dis) + "\n\n" + \
                  "棋子位置为:[%d,%d]" % checker_loc + "\n" + \
                  "目标位置为:[%d,%d]" % (target_loc[0], target_loc[1]) + "\n" + \
                  "跳跃距离为:%f" % distance + "\n" + \
                  "转换系数k为:%f" % k_array._get_k(distance) + "\n"
        _log(log_con, "EVENT")
        _jump(distance, k_array)
        _log("跳完！", "EVENT")

        k_array._save_np_array_data()
        time.sleep(sleep_duration)
        i += 1

    k_array._save_np_array_data()


def _init_log():  # 在log中加标记以便与历史纪录区分
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
    print("开始截屏...", end="")
    _cmd('adb shell screencap -p /sdcard/%s.png' % str(name))
    _cmd('adb pull /sdcard/%s.png ./temp' % str(name))
    print("完成!")


def _fix_cen_loc(cen_loc, fix_kx, fix_ky):
    k = min(fix_kx, fix_ky)
    return [cen_loc[0] * k, cen_loc[1] * k]


def my_int(num):
    return int(round(num))


def find_between(args, con):
    left_index = con.find(args[0])
    right_index = con.rfind(args[1])
    if (left_index + 1) * (right_index + 1):
        return con[left_index + len(args[0]):right_index]
    return


def init_files(cla):  # 检查工作环境和文件存在性，如果不存在报错
    for file_name in cla.work_files:
        cla.work_files[file_name] = os.path.join(cla.work_path,
                                                 cla.work_files[file_name])
        if not os.path.exists(cla.work_files[file_name]):
            os.mknod(cla.work_files[file_name])
    for file_name in cla.work_dirs:
        cla.work_files[file_name] = os.path.join(cla.work_path,
                                                 cla.work_files[file_name])
        if not os.path.exists(cla.work_files[file_name]):
            os.mkdir(cla.work_files[file_name])


class K_array:
    def __init__(self, img_rgb, tan, cos, sin, fix_kx, fix_ky, k0=2.34, k_array_sep=5, fix_step=1,
                 k_nparray_filename=r"np_array_data.txt"):
        self.k0 = k0
        self.work_path = os.path.join(os.path.dirname(__file__), "K_array")
        self.work_dirs = {"fig": "fig",
                          "data": "data"}
        self.init_files()

        self.fix_step = fix_step
        self.k_nparray_filename = os.path.join(self.work_dirs["data"], k_nparray_filename)
        self.k_array_sep = k_array_sep

        self.np_array_data = self._get_np_array_data(k0, img_rgb, tan, cos, sin, fix_kx, fix_ky)
        self._prepare_plot()

    def init_files(self):  # 检查工作环境和文件存在性，如果不存在报错
        for file_name in self.work_dirs:
            self.work_dirs[file_name] = os.path.join(self.work_path,
                                                     self.work_dirs[file_name])
            if not os.path.exists(self.work_dirs[file_name]):
                os.mkdir(self.work_dirs[file_name])

    def _prepare_plot(self):
        # 准备画k_array图
        self.max_num = -1
        fig_list = os.listdir(self.work_dirs["fig"])
        for f in fig_list:
            num_t = int(find_between(["_", "."], f))
            if num_t > self.max_num:
                self.max_num = num_t

        self.x_d = range(0, self.np_array_data[0].shape[0] * self.k_array_sep, self.k_array_sep)
        plt.figure(figsize=(20, 3))

    def _get_np_array_data(self, k0, img, tan, cos, sin, fix_kx, fix_ky):
        # 准备k数据
        self.fix_kk = _cal_dis((0, 0), (1, 1), cos, sin) / _cal_dis((0, 0), (fix_kx, fix_ky), cos, sin)
        _log("全局分辨率对应k转换系数为%f"%self.fix_kk,"EVENT")
        if os.path.exists(self.k_nparray_filename):
            np_array_data = np.loadtxt(self.k_nparray_filename)
        else:
            fake_p = (img.shape[1] / self.k_array_sep, img.shape[1] * tan / self.k_array_sep)
            k_num = my_int(_cal_dis((0, 0), fake_p, cos, sin))

            np_array_data = np.array([[-1] * k_num, [-2] * k_num, [k0] * k_num])  # 极小值，极大值，当前值
            for j in range(0, np_array_data[0].shape[0]):
                np_array_data[2][j] += (492 - j * self.k_array_sep) / 1530

        return np_array_data

    def _save_np_array_data(self):
        np.savetxt(self.k_nparray_filename, self.np_array_data)
        shutil.copyfile(self.k_nparray_filename, self.k_nparray_filename + "BAK")

    def _get_k(self, distance):
        i = my_int(distance / self.k_array_sep)
        return self.np_array_data[2][i]*self.fix_kk

    def _fix_k(self, distance, err_dis):

        i = my_int(distance / self.k_array_sep)
        self.pre_k = self.np_array_data[2][i]
        if distance < 10:
            return
        if err_dis >= 0:
            self.np_array_data[0][i] = self.np_array_data[2][i]
        else:
            self.np_array_data[1][i] = self.np_array_data[2][i]

        if abs(err_dis) > 10 or self.np_array_data[0][i] > self.np_array_data[1][i] or self.np_array_data[0][i] < 0:
            self.np_array_data[2][i] += err_dis * self.fix_step / (distance*self.fix_kk)
        else:
            self.np_array_data[2][i] = (self.np_array_data[0][i] + self.np_array_data[1][i]) / 2

    def _plot_and_save(self):
        plt.axis(
            [0, self.x_d[-1], max(np.min(self.np_array_data[2])*self.fix_kk, 0), min(np.max(self.np_array_data[2])*self.fix_kk, 3 * self.k0)])
        plt.plot(self.x_d, self.np_array_data[0]*self.fix_kk, "r.")
        plt.plot(self.x_d, self.np_array_data[1]*self.fix_kk, "b.")
        plt.plot(self.x_d, self.np_array_data[2]*self.fix_kk, "g.")
        self.max_num += 1
        plt.savefig(os.path.join(self.work_dirs["fig"], "fig_%d.png") % (self.max_num))
        plt.clf()


def _cmd(cmd_str):
    p = os.popen(cmd_str)
    p.read()
    # print(cmd_str,r.read(),sep="\n")


def _jump(distance, k_array):
    k = k_array._get_k(distance)
    press_time = my_int(distance * k)
    cmd_str = 'adb shell input swipe 320 1000 320 1000 ' + str(press_time)
    _cmd(cmd_str)


def _cal_dis(P1, P2, cos, sin):
    return _cal_dis_s(P1, P2, cos, sin)


def _cal_dis_n(P1, P2):
    return ((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2) ** 0.5


def _cal_dis_s(P1, P2, cos, sin):
    return ((P1[0] - P2[0]) ** 2 / (2 * (cos ** 2)) + (P1[1] - P2[1]) ** 2 / (2 * (sin ** 2))) ** 0.5


def _get_target_loc(img_rgb, checker_loc, checker_LT_loc, cen_loc, tan, cos, sin, search_begin_row=400):
    top_y = None
    target_loc = None
    if checker_loc[0] < img_rgb.shape[1] / 2:
        b = checker_LT_loc[0] + 51
        e = img_rgb.shape[1]
    else:
        b = 0
        e = checker_LT_loc[0]
    for i in range(search_begin_row, img_rgb.shape[0]):
        h = img_rgb[i][b:e]
        f_c = h[0]
        r = np.where(h != f_c)
        if r[0].shape[0]:
            top_y = i
            x = np.mean(r[0]) + b
            det_y = tan * abs(x - cen_loc[0]) - abs(top_y - cen_loc[1])  # 利用绝对中心找到偏移量
            y = top_y + abs(det_y)
            target_loc = (my_int(x), my_int(y))
            break

    # 计算上一次的跳跃目标
    pre_target_loc = (my_int(2 * cen_loc[0] - target_loc[0]), my_int(2 * cen_loc[1] - target_loc[1]))

    # 检查上一次跳跃与完美的偏差
    err_dis = _cal_dis(pre_target_loc, checker_loc,cos,sin)
    if checker_loc[1] - pre_target_loc[1] < 0:
        err_dis = -err_dis

    return top_y, target_loc, pre_target_loc, err_dis


def _save_outputimg(img_rgb, top_y, target_loc, pre_target_loc, checker_loc, name="output"):
    # 将图片输出以供调试
    output_rgb = img_rgb
    output_rgb = cv2.circle(output_rgb, (target_loc[0], top_y), 10, (255, 255, 255), -1)
    output_rgb = cv2.circle(output_rgb, target_loc, 10, (0, 0, 255), -1)
    output_rgb = cv2.circle(output_rgb, checker_loc, 10, (0, 255, 0), -1)
    output_rgb = cv2.circle(output_rgb, pre_target_loc, 10, (255, 0, 0), -1)
    cv2.imwrite(r'temp\%s.png' % name, output_rgb)
    return output_rgb


if __name__ == "__main__":
    _main()
