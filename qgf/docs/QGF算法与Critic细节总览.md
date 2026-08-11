# QGF 算法与 Critic 细节总览

这份文档整理我们目前 SmolVLA + QGF + Q critic 的整体逻辑。重点包括：

```text
SmolVLA 本身怎么出动作
QGF 怎么引导
beta 是什么
critic 输入有哪些向量
有没有视觉和语言
MLP critic 和 Transformer critic 区别
训练方式是不是 IQL
uncertainty gate 是什么
progress score 是什么
retrieval 是什么
counterfactual action critic 是什么
我们做过哪些微调
```

## 1. 整体流程

没有 QGF 的 baseline SmolVLA：

```text
图像 + 机械臂状态 + 语言指令
    -> SmolVLA
    -> 未来一段 action chunk
    -> 机器人执行
    -> 得到成功/失败
```

加入 QGF 后：

```text
图像 + 机械臂状态 + 语言指令
    -> SmolVLA 生成候选 action chunk 的去噪过程
    -> Q critic 给候选 action chunk 打分
    -> 对 action chunk 求梯度
    -> 修改 SmolVLA 的 flow velocity
    -> 得到被 Q 引导后的 action chunk
    -> 机器人执行
```

一句话：

```text
SmolVLA 负责“提出动作”，Q critic 负责“评价动作”，QGF 负责“沿着 Q 分数更高的方向轻轻推一下动作”。
```

## 2. SmolVLA 输出的是什么

SmolVLA 不是只输出一步动作，而是输出一个未来动作片段：

```text
action_chunk
```

在我们 LIBERO 代码里，critic 常见的动作维度是：

```text
action_dim = 7
```

动作 horizon 常见是：

```text
action_horizon = 50
```

所以一个 action chunk 可以理解成：

```text
未来 50 步，每一步 7 维动作
```

张量形状类似：

```text
[batch, 50, 7]
```

这 7 维一般对应机械臂末端/关节控制与夹爪控制，具体含义取决于 LeRobot/环境 postprocessor 的动作定义。

## 3. SmolVLA 的 Flow 约定

我们当前固定的 SmolVLA flow 约定是：

```text
x_t = t * noise + (1 - t) * action
v_t = noise - action
```

推理从：

```text
t = 1
```

开始，也就是纯噪声附近。

最后走到：

```text
t = 0
```

也就是干净 action。

在任意中间时刻，可以估计当前干净动作：

```text
a_hat = x_t - t * v_t
```

这里：

```text
x_t   = 当前 noisy action chunk
v_t   = SmolVLA 当前预测的 velocity
t     = 当前 denoising 时间
a_hat = 估计出来的 clean action chunk
```

## 4. Vanilla QGF 的数学逻辑

critic 是一个函数：

```text
Q(obs_features, action_chunk, task_features) -> scalar
```

它输出一个标量分数，越高表示这个 action chunk 越可能成功。

QGF 在每个 denoising step 做：

```text
1. SmolVLA 给出 velocity:
   v_t

2. 根据 x_t 和 v_t 估计 clean action:
   a_hat = x_t - t * v_t

3. critic 打分:
   q = Q(obs_features, a_hat, task_features)

4. 对 action 求梯度:
   g = grad_a Q(obs_features, a_hat, task_features)

5. 修改 velocity:
   v_guided = v_t - g / beta
```

因为当前 SmolVLA 是 reverse-time convention，所以这里是：

```text
v_guided = v_t - g / beta
```

这个符号非常关键。不是所有 flow policy 都是这个符号。

## 5. beta 到底是什么意思

公式是：

```text
v_guided = v_t - grad_Q / beta
```

所以：

```text
beta 越小 -> Q 引导越强
beta 越大 -> Q 引导越弱
```

比如：

```text
beta = 1  -> grad_Q 完整加入，力度很强
beta = 4  -> grad_Q 除以 4，力度中等
beta = 20 -> grad_Q 除以 20，力度较弱
```

所以 `beta=4` 不是说 Q 占 1/4 的决策权，也不是说 Q 乘以 4，而是说：

```text
Q 的 action gradient 被除以 4 后加到 flow velocity 里。
```

## 6. 有没有用 Jacobian？有没有 BPTT？

我们当前实现没有对整个 SmolVLA 去噪网络做反向传播，也没有显式求 denoiser Jacobian。

当前代码做的是：

```text
detach x_t 和 v_t
只让 a_hat requires_grad
计算 Q(a_hat)
求 grad_a Q
用这个梯度改 v_t
```

这意味着：

```text
不会训练 SmolVLA 本体；
不会 backprop through time 穿过整个去噪过程；
不会求 SmolVLA 网络关于动作的完整雅可比矩阵；
只求一次 critic 对 action 的梯度。
```

这就是它轻量的地方。

但这不等于完全解决 OOD。因为如果 critic 本身不准，或者没看到关键视觉信息，它的梯度仍然可能把动作往错误方向推。

## 7. 当前 Q critic 的输入向量

当前代码里，QGF runtime 传给 critic 的主要是：

```text
obs_features = observation.state
action_chunk = estimated clean action chunk
task_features = optional language feature
```

具体代码位置：

```text
src/guided_action_flow/policies/smolvla_qgf.py
src/guided_action_flow/guidance/qgf.py
```

因此当前 critic 主要输入是：

```text
机械臂/预处理状态向量
未来动作 chunk
可选任务语言向量
```

当前 critic 没有直接输入：

```text
raw RGB image
front camera embedding
wrist camera embedding
object detection boxes
object-relative pose
```

## 8. 当前有没有视觉和语言

分开看：

| 模块 | 是否有视觉 | 是否有语言 |
| --- | --- | --- |
| SmolVLA base policy | 有 | 有 |
| 当前 Q critic MLP | 没有直接 RGB | 可选 hashed tokens / VLM hidden |
| 当前 QGF 引导 | 通过 SmolVLA 间接受视觉，但 critic 本身不直接看图 | 可选 task feature |
| 未来 vision-aware critic | 应该有 image embedding | 应该有 language embedding |

所以最准确表述：

```text
当前 SmolVLA 有视觉，但当前 Q critic 不是视觉 critic。
```

## 9. MLP Critic 架构

文件：

```text
src/guided_action_flow/critics/action_chunk_critic.py
```

输入拼接：

```text
flat_obs_features
+ optional proprio
+ optional task_features
+ flat_action_chunk
```

然后走 MLP：

```text
Linear
SiLU
Linear
SiLU
...
Linear -> scalar Q
```

常见配置：

```text
hidden_dim = 512
depth = 3
```

优点：

```text
简单
快
容易训练
数据少时不容易太夸张地过拟合
```

缺点：

```text
把 action chunk 直接 flatten；
对时间顺序建模不够自然；
表达能力比 Transformer 弱；
如果输入没有图像，它仍然看不到视觉。
```

## 10. Transformer Critic 架构

文件：

```text
src/guided_action_flow/critics/transformer_action_chunk_critic.py
```

它把不同东西变成 token：

```text
[CLS]
obs token
optional proprio token
optional task token
action token 0
action token 1
...
action token H-1
```

每一步 action 通过线性层变成一个 token：

```text
action_t -> Linear(action_dim, d_model)
```

然后输入 Transformer Encoder：

```text
TransformerEncoder -> CLS token -> MLP head -> scalar Q
```

常见配置：

```text
d_model = 256
num_layers = 3
num_heads = 4
dropout = 0.1
```

优点：

```text
更适合 action chunk 序列；
能建模第 1 步动作和第 50 步动作之间的关系；
比 MLP 更有表达能力；
更接近学长说的 transformer critic 方向。
```

缺点：

```text
更吃数据；
更需要调参；
如果仍然没有图像输入，它只是更强的 state/action critic，不会自动变成视觉 critic。
```

## 11. Task Language Feature 是什么

我们做过/实现过两种语言特征。

### 11.1 Hashed Token Feature

文件：

```text
src/guided_action_flow/training/task_features.py
```

逻辑：

```text
task instruction -> tokenizer tokens -> hash 到固定维度 -> 归一化向量
```

比如：

```text
task_feature_dim = 128
```

优点：

```text
便宜；
不用再跑大模型；
训练和推理都简单。
```

缺点：

```text
语义比较弱；
不能很好理解句子意思；
更多像 bag-of-words。
```

### 11.2 VLM Hidden Feature

逻辑：

```text
task tokens -> SmolVLA/VLM text stack -> hidden states -> mean pooling -> task feature
```

优点：

```text
语义更强；
和 SmolVLA 自己的语言模型更对齐。
```

缺点：

```text
更慢；
维度和 runtime 必须严格对齐；
它仍然主要是语言特征，不是图像特征。
```

## 12. Success-To-Go Target

最基础的训练标签是：

```text
success_to_go
```

如果未来会成功：

```text
target_t = gamma^(距离未来成功还有多少步)
```

如果未来不会成功：

```text
target_t = 0
```

成功之后：

```text
target_t = 1
```

这个方法只需要 episode 最终 success/fail，不需要每帧人工标注。

缺点：

```text
失败轨迹大部分 target 都是 0；
差一点成功和完全失败区分不开。
```

## 13. Progress Score Target

为了让失败轨迹也有信息，我们讨论过：

```text
target = 0.7 * success_to_go + 0.3 * progress_score
```

这里：

```text
success_to_go: 0 到 1
progress_score: 0 到 1
target: 0 到 1
```

progress_score 可以表示任务进度：

```text
接近物体了吗？
夹爪对准了吗？
抓住了吗？
移动到目标上方了吗？
放下了吗？
```

这样即使最后失败，也可以有：

```text
0.2 / 0.4 / 0.6
```

而不是全部 0。

## 14. Retrieval Progress 是什么

retrieval 的意思是“检索相似状态”。

我们在成功轨迹里建立一个状态库：

```text
成功 rollout 的 state -> 对应 success_to_go
```

对某个新状态：

```text
找到 K 个最相似的成功状态
取这些成功状态的 value 平均
作为 progress estimate
```

然后：

```text
target = 0.7 * success_to_go + 0.3 * retrieval_progress
```

优点：

```text
不需要人工逐帧标 progress；
失败轨迹也能获得非零进度标签；
实现相对容易。
```

缺点：

```text
如果检索只基于 robot state，它仍然可能不看图像；
物体位置变了但机械臂姿态相似时，retrieval 可能误判。
```

## 15. Uncertainty Gate 是什么

uncertainty gate 通常和 ensemble critic 一起用。

比如训练 3 个 critic：

```text
Q1, Q2, Q3
```

同一个 action chunk 输入进去：

```text
q_mean = mean(Q1, Q2, Q3)
q_std = std(Q1, Q2, Q3)
```

如果 3 个 critic 分数很一致：

```text
q_std 小 -> 说明比较确定
```

如果 3 个 critic 分歧很大：

```text
q_std 大 -> 说明不确定
```

gate 公式：

```text
gate = exp(-uncertainty_scale * q_std)
gate = clamp(gate, min_gate, 1.0)
```

最后：

```text
guided_grad = grad_Q * gate
```

意义：

```text
critic 确定时，多引导；
critic 不确定时，少引导；
防止 OOD 情况下乱推。
```

如果：

```text
uncertainty_scale = 0
```

那么：

```text
gate = 1
```

也就是 uncertainty gate 关闭。

## 16. Gradient Clip 是什么

Q gradient 可能很大，所以我们可以裁剪：

```text
grad = grad * min(1, grad_clip_norm / ||grad||)
```

作用：

```text
防止 guidance 太猛；
防止动作被 Q 一下子推飞；
让 beta 调参更稳定。
```

## 17. IQL 到底是什么，我们现在是不是 IQL

严格说，当前 `scripts/train_critic.py` 主要是监督式 Q 回归：

```text
MSE(Q(state, action_chunk), target)
```

target 可以是：

```text
success_to_go
progress_blend
retrieval_progress_blend
```

这叫 Q-like critic，但不等于完整 IQL。

完整 IQL 通常有：

```text
Q(s, a)
V(s)
advantage = Q(s, a) - V(s)
expectile regression 训练 V
Bellman target 训练 Q
advantage-weighted behavior cloning 训练 policy，可选
```

所以如果我们严谨汇报：

```text
当前代码实现的是 QGF-compatible supervised action-chunk critic；
IQL/Transformer-IQL 是我们正在对齐或准备复现的更强离线 critic 训练方向。
```

不要把现在所有东西都说成完整 IQL。

## 18. Counterfactual Action Critic 是什么

普通 rollout 数据是：

```text
状态 S -> 模型实际做了动作 A -> 最后成功/失败
```

问题是，我们不知道：

```text
如果同一个状态 S 下做动作 B，会不会更好？
```

counterfactual action critic 想解决这个问题。

在模拟器里，我们可以保存同一个状态 S：

```text
保存 MuJoCo / robosuite simulator state
```

然后恢复同一个 S，分别执行不同 action chunk：

```text
S + action A -> outcome_A
S + action B -> outcome_B
S + action C -> outcome_C
```

如果：

```text
outcome_A > outcome_B
```

就训练：

```text
Q(S, A) > Q(S, B)
```

这个叫 ranking loss。

## 19. Counterfactual Mixed Critic 的训练目标

文件：

```text
scripts/train_counterfactual_mixed_critic.py
```

它混合两类训练：

### 普通 rollout MSE

```text
MSE(Q(s, a), success_to_go)
```

### 反事实 ranking loss

```text
softplus(-(Q(s, good_action) - Q(s, bad_action)))
```

总 loss：

```text
total_loss =
    mse_weight * mse_loss
  + ranking_weight * ranking_loss
  + cf_regression_weight * cf_regression_loss
```

意义：

```text
普通 MSE 学“这条轨迹最后成不成功”；
ranking loss 学“同一个状态下哪个动作更好”。
```

这比单纯 success-to-go 更接近真正 action critic。

## 20. 我们做过哪些微调

### 20.1 Vanilla QGF

只做：

```text
v_guided = v_t - grad_Q / beta
```

没有 progress score，没有 uncertainty gate，没有 counterfactual。

目的：

```text
先证明 Q 确实插进 SmolVLA denoising loop。
```

### 20.2 Beta Sweep

试不同 beta。

观察：

```text
beta 太小 -> guidance 太强，容易破坏原动作；
beta 太大 -> guidance 太弱，可能没效果；
最佳 beta 可能随任务、seed、critic 变化。
```

### 20.3 Gradient Clip

控制 Q gradient 大小，防止动作被推太多。

### 20.4 Multi-task Critic

不是只训练一个任务，而是多个 task 一起训练。

目的：

```text
让 critic 学 task family 里的共性；
尝试泛化到没训练过的任务。
```

问题：

```text
如果没有视觉，多任务也可能只是学到动作/机械臂状态先验。
```

### 20.5 Task Language Feature

加入任务语言向量，让 critic 区分不同指令。

包括：

```text
hashed token feature
VLM hidden feature
```

### 20.6 Retrieval Progress Target

通过检索成功轨迹里的相似状态，给失败轨迹提供非零 progress。

### 20.7 Uncertainty Gate / Ensemble

训练多个 critic，当它们分歧大时减少 QGF 引导。

### 20.8 Transformer Critic

把 action chunk 当序列建模，而不是 flatten 后丢给 MLP。

### 20.9 Counterfactual Branch Data

在同一个 simulator state 下跑 A/B/C 不同动作，训练 ranking。

### 20.10 Mixed MSE + Ranking

把 success-to-go 回归和 counterfactual ranking 结合。

## 21. 当前最大问题

当前最大问题不是 beta，也不只是 MLP/Transformer。

最大问题是：

```text
critic 输入信息不够。
```

当前 critic 主要看：

```text
state + action chunk + optional language
```

但真实机器人操作里很关键的是：

```text
物体在哪里；
夹爪是否对准；
目标容器在哪里；
瓶子是否倾斜；
当前视角下有没有遮挡；
背景/光照有没有变化。
```

这些主要来自视觉。

所以后面如果想做更有论文价值的点，最自然的是：

```text
vision-aware counterfactual action critic
```

## 22. 推荐下一版架构

建议做一个视觉版 Transformer critic：

```text
Q(image_front, image_wrist, robot_state, language, action_chunk) -> scalar
```

Token 设计：

```text
[CLS]
front_image_token
wrist_image_token
robot_state_token
language_token
action_token_0
action_token_1
...
action_token_H
```

训练目标：

```text
success-to-go MSE
+ progress score / retrieval progress
+ counterfactual ranking loss
+ uncertainty ensemble, optional
```

最终对照：

```text
SmolVLA baseline
SmolVLA + state-only QGF
SmolVLA + vision-aware QGF
SmolVLA + vision-aware counterfactual QGF
```

如果 vision-aware 版本在物体布局、视角变化、真机不同初始位置下明显更稳，这就是一个很清楚的创新点。

## 23. 可以对导师这样讲

可以总结成：

```text
我们目前的 QGF 是在 SmolVLA flow denoising 过程中加入 Q critic 的 action-gradient guidance。
SmolVLA 先预测 velocity，我们估计 clean action chunk，然后用 Q critic 对这个 action chunk 打分，
再用 grad_a Q / beta 修改 velocity。当前实现不会反传 SmolVLA，也不求完整 Jacobian。

当前 critic 已经尝试了 MLP、Transformer、multi-task、task language feature、
retrieval progress target、uncertainty gate 和 counterfactual ranking。
但目前 critic 主要输入是 observation.state 和 action chunk，没有直接输入 RGB 图像，
所以它更像轨迹进度/动作先验 critic，而不是真正视觉 grounding 的 action critic。

下一步真机采数据时，我们应该保存每帧图像、机器人状态、语言、预测 action chunk、
真实执行 action 和最终 success/fail，并训练 vision-aware counterfactual action critic，
让 Q 不只是判断这条动作像不像成功轨迹，而是判断当前画面下这个动作是否真的更好。
```

