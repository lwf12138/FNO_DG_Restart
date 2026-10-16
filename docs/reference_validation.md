> 历史技术记录。当前状态和可执行命令以根目录 README 与 docs/RUNNING.md 为准；文中“本轮未执行”等措辞指原交付时点。

# 本次问题与精确解核验

## 1. 问题、参数和初值

在 $0\le x\le L=2\pi$、$0\le t\le1$ 上求解无源系统

$$
\epsilon_b E_t=H_x-P_t,\qquad \mu H_t=E_x,
\qquad \tau^\alpha\,{}^CD_t^\alpha P+P=\Delta\epsilon E,
$$

其中 $\epsilon_b=2,\mu=1,\Delta\epsilon=1$。分数阶导数为从 $t=0$ 开始的 Caputo 导数；初始极化及更早的材料记忆取零。PEC 条件为 $E(0,t)=E(L,t)=0$。$H$ 的端点不另设零值。

每条轨迹内 $\alpha,\tau$ 固定，轨迹之间改变材料及初值：

$$E_0=\sum_{m=1}^3 a_m\sin(m\pi x/L),\qquad
H_0=\sum_{m=1}^3 b_m\cos(m\pi x/L),\qquad P_0=0.$$

$\alpha$ 为 $[0.3,0.7]$ 的8个等距值；$\tau$ 为 $[0.5,2]$ 的8个对数等距值。每个材料组分别抽8组初值，$a_m,b_m$ 独立服从 $U(-1,1)/m^2$。随机种子42。共64个材料组、512条完整轨迹；具体数值均在 `data/samples.csv`。材料重复8次是为了改变初值，并非重复完整样本。

此前的 $t^{10}$ 制造解及其外源不适用于这些新初值。本目录使用下面针对本次无源问题推导的解析解；旧标签没有参与生成。

## 2. 对应的模态解析解

均匀材料使三个空间模态相互独立。固定 $\kappa=m\pi/L$，令 $E=e(t)\sin\kappa x,H=h(t)\cos\kappa x,P=p(t)\sin\kappa x$，则

$$\epsilon_b e'=-\kappa h-p',\quad \mu h'=\kappa e,\quad
\tau^\alpha{}^CD_t^\alpha p+p=\Delta\epsilon e.$$

初值 $e(0)=a,h(0)=b,p(0)=0$。记

$$\chi(s)=\frac{\Delta\epsilon}{1+(\tau s)^\alpha},\quad
D(s)=\epsilon_b s^2+\frac{\kappa^2}{\mu}+s^2\chi(s).$$

其精确拉普拉斯表示为

$$\widehat e(s)=\frac{\epsilon_b sa-\kappa b}{D(s)},\qquad
\widehat h(s)=\frac b s+\frac{\kappa\widehat e(s)}{\mu s},\qquad
\widehat p(s)=\chi(s)\widehat e(s).$$

复数幂采用主分支。这个表示来自连续方程，与 DG 网格及时间步长无关。

实际参考值用分数幂级数求值。令 $\lambda=\tau^{-\alpha}$、$A=(1+\Delta\epsilon/\epsilon_b)\lambda$、$B=\kappa^2/(\epsilon_b\mu)$、$C=B\lambda$，定义

$$c_{00}=1,\quad c_{ij}=-Ac_{i-1,j}-Bc_{i,j-1}-Cc_{i-1,j-1},\quad
d_{ij}=c_{ij}+\lambda c_{i-1,j},$$

负下标系数为零。用 $S(c,r)=\sum_{i,j\ge0}c_{ij}t^{i\alpha+2j+r-1}/\Gamma(i\alpha+2j+r)$，有

$$
e=aS(d,1)-\frac{\kappa b}{\epsilon_b}S(d,2),\qquad
h=b+\frac{\kappa a}{\mu}S(d,2)-BbS(d,3),
$$
$$p=\Delta\epsilon\lambda\left[aS(c,\alpha+1)-\frac{\kappa b}{\epsilon_b}S(c,\alpha+2)\right].$$

代码在 $t=0$ 直接使用规定初值。正时刻比较 $(i_{max},j_{max})=(192,20)$ 与 $(256,28)$ 的截断结果，并另用24、32项固定 Talbot 逆变换交叉验证，容许差异 $5\times10^{-9}$。Talbot 公式参见 [mpmath 官方说明](https://mpmath.org/doc/current/calculus/inverselaplace.html#fixed-talbot-algorithm)；这里只实现普通双精度公式，没有调用 mpmath 或100位计算。完整64个材料组的实测结果在 `checks/reference_evaluation.json`。

数学解的解析表示是精确的，计算机给出的参考值仍有截断和舍入误差。这些交叉检查是数值证据，不是严格误差界。

## 3. DG 标签如何产生

每条轨迹实际启动一次 C++ DG–RadauIIA-3 RK-CQ 求解，取 $N=64,k=2,\Delta t=0.0005$，共2000步。使用 `double`，保留完整分数阶历史；block-FFT 只加速历史求和。相同 $\alpha$ 等设置下复用 CQ 权重，不复用轨迹解。初始 $E,H$ 作逐单元 $L^2$ 投影，$P$ 为零。因此离散初始帧与连续初值可能有小量投影误差。

原内核只增加两处接口：接受新的初值函数、关闭制造解外源。空间装配、RK、CQ 和端点更新保持原样；差异可查 `checks/cole_cole.cpp.patch`。原双精度文件保存在 `vendor/base_double`，来源及校验值记录在 `checks/source_manifest.json`。

每单元二次多项式以首一 Legendre 基 $1,\xi,\xi^2-1/3$ 正确求值。65个等距采样点恰在 $N=64$ 的界面上：内部界面取两侧迹的算术平均，两端取域内迹。PEC 通过数值通量实施，采样端点不强行修改为零。误差核验使用整个单元内部的多项式，不用这些界面平均值估计误差。

## 4. 逐条验证及适用范围

每条轨迹的11帧、每个场分别计算

$$\|u_h-u_{exact}\|_{L^2(0,L)},\qquad
\frac{\|u_h-u_{exact}\|_{L^2}}{\|u_{exact}\|_{L^2}}.$$

每单元24点 Gauss 积分；解析场范数还与模态正交性公式 $\sqrt{\pi\sum_m u_m(t)^2}$ 对照。合格条件为

$$\|u_h-u_{exact}\|_{L^2}\le10^{-6}+10^{-3}\|u_{exact}\|_{L^2}.$$

精确场范数为零时只报告绝对误差，相对误差存 NaN 并以 `relative_defined=False` 标记。这不属于求解产生非有限数。物理场和全部 DG 系数必须有限。

先做5个独立预检案例，其中3例另算 $N=128,k=2,\Delta t=0.00025$，两套都与解析解对照；预检不计入512条。再逐条核验完整数据。结果在 `checks/preflight_summary.json`、`checks/l2_errors.csv` 和 `checks/validation_summary.json`。

这证明本次指定初值、材料范围和已保存时刻下的误差符合筛选标准，不能保证任意新问题无误，也不保证未保存时刻的最大误差。512条仅用于后续学习流程和有限范围实验；同材料组全部初值、全部时间帧属于同一个划分，尚未训练网络或评估其泛化能力。

## 5. 输出和可追溯性

`runs/sample_XXXX/` 保留每条轨迹的原始系数 `coefficients.f64`（小端 float64，形状 `[11,3,64,3]`）、65点输出、运行命令、日志、参数和核验结果。系数最后一维是三个首一 Legendre 系数，不是三个空间节点。

`data/dataset.npz` 的 `fields` 是 DG 数值参考解标签，`exact_fields` 仅用于核验。`initial_fields` 是给定连续初值的65点采样；`dg_initial_fields` 是 DG 投影初始帧。`initial_coefficients` 为 `[样本,E/H,模态]`，三模态按1、2、3排列；`alpha,tau,sample_id,material_id,split` 均按样本对应。所有场保留原始物理尺度。

随机划分48/8/8个材料组，得到384/64/64条训练/验证/测试轨迹。未来若需要归一化，其统计量只允许从训练集合计算。测试集合在这里仅进行标签数值质量核验，不参与任何网络调参。
