# 许可与来源

项目自有代码按维护者选定的 [MIT License](LICENSE) 发布。许可证标准文本来源为 [Open Source Initiative](https://opensource.org/license/mit)。

数值内核来源为用户已有工程的双精度实现，保留在 vendor/base_double 与独立验收的原始源码快照中；文件来源与最小接口扩展记录见 docs/PROVENANCE.md。项目许可证不替代第三方依赖各自的许可证。

神经网络依赖官方 neuraloperator 和 PyTorch，数值与绘图依赖 NumPy、SciPy、Matplotlib，C++ 编译依赖 Boost；这些依赖由包管理器安装，不包含在本仓库中，其许可证由各上游项目维护。
