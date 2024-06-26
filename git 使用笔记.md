# git 使用笔记



[toc]

![git工作原理](img\git工作原理.png)

##  一.安装及配置 

###  1.1下载

+ [git下载官网](https://git-scm.com/download)

+ 下载完成后双击安装，自定义安装目录，勾选建立桌面快捷方式

+ 安装完成后桌面空白处右键，弹出列表中出现 ***Open Git Bash here***及 ***Open Git GUI here***

  *（如果是**win 11**，则需要按着**Shift**的同时右键）*，***Bash***为命令行页面，***GUI***为图形化页面，我们一般使用 ***Open Git Bash here***。

### 1.2配置

+ *首次使用必须配置邮箱，密码！！！*配置命令如下：


```bash
git config --global user.name "YourName"
git config --global user.email "YourEmail@xxx"
```

+ 验证配置情况：
  
```bash
git config --global user.name
git config --global user.email
```

## 二.使用流程

###  2.1仓库初始化

+ 首先在要创建仓库的文件夹右键 Open Git Bash here，或者先打开git，然后 `cd` 到仓库的地址

```bash
cd /路径
git init
```

+ 初始化后文件夹中出现一个隐藏文件夹.git，打开 *文件管理器* — *查看* — *显示* — *隐藏的项目* 即可查看，加`all`即查看所有文件

```bash
ll -a
```
+ 连接github仓库

  1. 通过 http 方式连接
	`git remote add origin <仓库http地址>`

  2. 通过 ssh 密钥连接(速度更快，更安全)
	+ 生成 ssh key (终端中输入)
		`ssh-keygen -t rsa -C "youremail@example.com`

	+ 添加到 github
		`cat ~/.ssh/id_rsa.pub`

	+ 复制输出内容，打开 github，点击右上角头像，选择 `settings` -> `SSH and GPG keys` -> `New SSH key`，将复制的内容粘贴到 key 中，点击 `Add SSH key`。

	+ 然后同1步骤
		`git remote add origin <仓库ssh地址>`

### 2.2 原理

![git仓库连接](img\git仓库连接.png)

1. 要想提交更改到github中，首先要将工作区中（工作区即除***.gitignore***指定的文件之外的所有内容)要提交的内容添加到暂存区

```bash
# 想要提交指定的文件，即可输入
git add 指定文件路径
# 多项内容同时需要提交时，可输入
git add .
```

2. 将暂存区内容提交到本地仓库，并加上修改说明

```bash
git commit -m "注释"
```

3. 本地仓库内容上推github
```bash
git push origin "分支名"
```

4. github仓库内容下拉更新(本地会被覆盖)
```bash
git pull origin "分支名"
```

5. 若想查看修改的状态和更改记录或者想回退版本

```bash
# 查看当前工作区及暂存区状态
git status
# 查看当前更改记录及commitID
git log 
# 回退版本(分支)
git reset --hard commitID
# 查看历史更改记录(包括已删除的分支)
git reflog
```

6. 使用`git add .`提交时若想忽略某些文件

```bash
# 创建.gitignore文件(固定名称)
touch .gitignore
# 编辑.gitignore内容
vim .gitignore
# 想忽略哪个填哪个即可，可使用正则表达式
```

7. 创建、切换、删除及合并分支

```bash
# 想要创建一个分支
git branch 分支名
# 当前在主线，要切换到分支
git checkout 分支名
# 若想创建一个当前不存在的分支并切换过去
git checkout -b 分支名
# 要删除某一分支(不能删除当前分支)
git branch -d 分支名 # -d需要检查，-D强制删除
# 当前在线支，要将分支合并到主支
git merge 分支名
```
