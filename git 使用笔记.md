# git 使用笔记



[toc]

![git工作原理](C:\Users\c3322\Desktop\git工作原理.png)

###  1.安装及配置 

> ####  1.1下载

+ [GIT下载官网](https://git-scm.com/download)

+ 下载完成后双击安装，自定义安装目录，勾选建立桌面快捷方式

+ 安装完成后桌面空白处右键，弹出列表中出现 ***Open Git Bash here***及 ***Open Git GUI here***

  *（如果是**win 11**，则需要按着**Shift**的同时右键）*，***Bash***为命令行页面，***GUI***为图形化页面，我们一般使用 ***Open Git Bash here***。



> #### 1.2配置

+ *首次使用必须配置邮箱，密码！！！*配置命令如下：

  ```bash
  git config --global user.name "YourName"`
  git config --global user.email "YourEmail@xxx"`
  ```
  
+ 验证配置情况：
  
	```bash
  git config --global user.name`
	git config --global user.email`
  ```
  
### 2.使用流程

> ####  2.1仓库初始化

+ 首先在要创建仓库的文件夹右键 Open Git Bash here，或者先打开git，然后 `cd` 到仓库的地址
  ```bash
  cd /路径
  git init
  ```

+ 初始化后文件夹中出现一个隐藏文件夹.git，打开 *文件管理器* — *查看* — *显示* — *隐藏的项目* 即可查看，加`all`即查看所有文件

  ```bash
  ll -a
  ```

> #### 2.2 原理





  

  

  

