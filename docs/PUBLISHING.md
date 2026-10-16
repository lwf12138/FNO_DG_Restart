# 发布操作

本目录是准备上传的项目；原始 FNO_DG_Restart 保留在本地。没有创建远程仓库或推送。

1. 阅读根目录 README 与 NOTICE，核对项目名称和来源说明。发布版已按你的选择加入 MIT License。
2. 运行 `python scripts/verify_release.py`。确认当前结果通过，且 README 的未通过项仍如实保留。
3. 执行 `git status --short` 和 `git diff --cached --stat` 查看待发布文件。虚拟环境、编译目录及新运行输出由 `.gitignore` 排除。
4. 创建一个空的 GitHub 仓库；复制 GitHub 提供的远程 URL 后执行以下命令。

```bash
git add .
git diff --cached --stat
git commit -m "Prepare Maxwell Cole-Cole FNO research prototype"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

若已经存在 origin，则先核对 `git remote -v`，不要重复添加。上述占位 URL 须替换为真实仓库地址。提交作者使用你自己的 Git 配置；本次整理不替你设定身份。

推荐仓库描述：`DG–RK-CQ reference data and FNO surrogate experiments for 1D Maxwell–Cole–Cole equations, with frozen-model generalization checks.`

首页应保留 G3/G4 失败结果和计时限制，不使用“任意初值可靠”“同精度加速500倍”或“已验证通用求解器”等描述。数据和选定模型体量适合当前的小型研究原型；不要提交原项目的全部历史运行。

压缩包与目录内容一致，压缩包不含 Git 元数据或本地虚拟环境。若从压缩包解压发布，先运行 `git init -b main`，再执行上面的提交与推送命令。
