# Wechat_JumpGame

【基于Python】自己写的一个微信跳一跳自动游戏程序。
最新测试破3万无压力。

* (不要超过3000分，分数太高服务器一刀切延迟封号）

* 全自动运行,失败自动重来
* 自动适应不同分辨率
* 自动调整各个参数误差，玩的越久表现越好~

不需要配置adb,不需要配置python环境

只需要

1. 安卓手机，开启开发者模式，允许开发者安全模式（允许模拟触屏）
2. 手机连接到电脑，手机上打开跳一跳游戏界面
3. 双击"运行.exe"

喝茶去吧!

如果一开始效果不好，
请适当调整K_array文件夹下的k0.txt里的数字（全局初始系数），直到第一次跳准【第一次跳的远就适当减小数字，跳得近就适当增加数字】(它可以继续不断自动微调)
![参数调整图](https://github.com/Haskely/Wechat_JumpGame/raw/master/readme_pic/fig.png)

--------------------------
如果想要研究代码，源代码在Source_Code文件夹里

运行代码需要配置python环境并运行require.bat以安装相应库依赖

从代码编译exe请运行buildexe.bat，编译出来的文件保存在Source_Code/dist文件夹内

![Image text](https://github.com/Haskely/Wechat_JumpGame/raw/master/readme_pic/screenshot.png)
![Image text](https://github.com/Haskely/Wechat_JumpGame/raw/master/readme_pic/output.png)
![Image text](https://github.com/Haskely/Wechat_JumpGame/raw/master/readme_pic/last_screenshot2.png)
![Image text](https://github.com/Haskely/Wechat_JumpGame/raw/master/readme_pic/last_screenshot3.png)
![Image text](https://github.com/Haskely/Wechat_JumpGame/raw/master/readme_pic/last_screenshot4.png)
