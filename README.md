## 🚀 CUDA 编程与算法研究

CUDA 编程在未来对于一个 **算法研究者** 🧑‍🔬 或者 **从业者** 👩‍💻 来说，几乎可以说是 <span style="color:orange; font-weight:bold;">基本功</span>。

为什么这么说呢？🤔  

- 🎯 **高性能计算必备**：深度学习、图像处理、科学计算等领域，GPU 并行加速已经成为核心驱动力。  
- ⚡ **理解更深层原理**：掌握 CUDA，可以让你更清晰地理解 **张量计算、并行优化** 背后的逻辑。  
- 🛠️ **工程与研究两不误**：不管是做科研论文，还是落地工业系统，CUDA 都能让你的算法在 <span style="color:green; font-weight:bold;">效率</span> 与 <span style="color:blue; font-weight:bold;">性能</span> 上脱颖而出。  
- 🔑 **行业竞争力**：未来的算法从业者如果不会 CUDA，就像是程序员不会数据结构和算法一样，几乎难以在高阶领域立足。  

✨ 换句话说，CUDA 编程不仅是一项技能，更是 <span style="color:red; font-weight:bold;">算法研究的必修课</span>。

---
### Day004 "LayerNorm CUDA Version"
```
./norm 
// Input A:
// 0.84 0.39 0.78 0.80 
// 0.91 0.20 0.34 0.77 
// 0.28 0.55 0.48 0.63 
// 0.36 0.51 0.95 0.92

// Output B:
// 0.76 -1.72 0.44 0.52 
// 1.21 -1.20 -0.74 0.73 
// -1.58 0.53 -0.05 1.10 
// -1.27 -0.68 1.05 0.91 
```

> ### 🔹 LayerNorm 公式
> 
> 对每一行向量 $\mathbf{x} = (x_1, x_2, \dots, x_H)$：
> 
> ### 均值：
> $$
> \mu = \frac{1}{H} \sum_{i=1}^{H} x_i
> $$
> 
> ### 方差：
> $$
> \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
> $$
> 
> ### 归一化：
> $$
> \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \epsilon = 10^{-5}
> $$

---
