## D3P: Dynamic Denoising Diffusion Policy via Reinforcement Learning

## Shu’ang Yu^12 , Feng Gao^1 , Yi Wu^13 , Chao Yu^1 †, Yu Wang^1 †,

(^1) Tsinghua University (^2) Shanghai AI Laboratory (^3) Shanghai Qi Zhi Institute
†Corresponding Authors
{yuchao, yu-wang}@mail.tsinghua.edu.cn
Figure 1: An overview ofDynamicDenoisingDiffusionPolicy (D3P).(a) Motivation:Robotic tasks involve actions of varying
criticality. Crucial actions, like object handover, have a greater impact on task success than routine actions.(b) Idea:Instead
of using a fixed number of denoising steps, D3P dynamically allocates more denoising steps to crucial actions.(c) Method:
D3P uses a base policyπθand a lightweight adaptorKω. The adaptor predicts the noise-level strides, which determines total
denoising steps for current action.(d) Results:D3P achieves a comparable success rate, while providing a 2.2×speed-up to a
normal fixed-step diffusion policy.
Abstract
Diffusion policies excel at learning complex action distribu-
tions for robotic visuomotor tasks, yet their iterative denois-
ing process poses a major bottleneck for real-time deploy-
ment. Existing acceleration methods apply a fixed number
of denoising steps per action, implicitly treating all actions
as equally important. However, our experiments reveal that
robotic tasks often contain a mix ofcrucialandroutineac-
tions, which differ in their impact on task success. Motivated
by this finding, we proposeDynamicDenoisingDiffusion
Policy(D3P), a diffusion-based policy that adaptively allo-
cates denoising steps across actions at test time. D3P uses a
lightweight, state-aware adaptor to allocate the optimal num-
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
ber of denoising steps for each action. We jointly optimize
the adaptor and base diffusion policy via reinforcement learn-
ing to balance task performance and inference efficiency.
On simulated tasks, D3P achieves an averaged 2.2×infer-
ence speed-up over baselines without degrading success. Fur-
thermore, we demonstrate D3P’s effectiveness on a physical
robot, achieving a 1.9×acceleration over the baseline.

## Introduction

```
Diffusion policies have demonstrated remarkable promise in
robotic visuomotor tasks (Janner et al. 2022; Pearce et al.
2023; Chi et al. 2023; Ze et al. 2024; Ma et al. 2024). By
casting action generation as a conditional denoising diffu-
sion process, they naturally model the full, often highly mul-
timodal distribution of feasible actions while retaining stable
```
# arXiv:2508.06804v1 [cs.RO] 9 Aug 2025


optimization dynamics (Chi et al. 2023). Concretely, action
sampling integrates the reverse-time stochastic differential
equation (SDE) that refines Gaussian noise into clean ac-
tions, following Ho, Jain, and Abbeel (2020) and Song et al.
(2020). This procedure entails dozens of denoising steps,
so diffusion policies typically run more slowly at inference
than one-shot generators based on GANs (Goodfellow et al.
2020), VAEs (Kingma, Welling et al. 2013), or autoregres-
sive models (Shafiullah et al. 2022; Zhao et al. 2023), which
limits their deployment in real-time control.
To address these challenges, previous work has sought to
accelerate inference through various strategies, such as re-
formulating the process as an ordinary differential equation
(ODE) to allow fewer sampling steps (Song, Meng, and Er-
mon 2020; Lu et al. 2022a,b), distilling the policy into a
single-step model (Prasad et al. 2024; Wang et al. 2024b),
or employing streaming techniques that use action history
as a prior (Høeg, Du, and Egeland 2024; Chen et al. 2025).
These acceleration methods for diffusion policies typically
use a uniform number of denoising steps per action, which
implicitly assumes that all actions are equally important for
task success.
However, our analysis of robotic tasks provides evidence
that contradicts this assumption. We observe a stark non-
uniformity in action criticality: task executions typically
consist of pivotalcrucial actionsthat largely determine
success, and more forgivingroutine actionswith only a
marginal impact. For instance, in aTransporttask, the
precise moment of object handover is crucial, whereas the
arm movements before and after have less impact on task
success. Treating all actions uniformly, therefore, incurs un-
necessary computational overhead and restricts the ability to
balance decision quality with inference efficiency. This in-
sight motivates a new design principle:Allocate denoising
steps adaptively: spend more computation on crucial ac-
tions, and less on routine ones—to balance precision and
efficiency.
In this paper, we proposeDynamicDenoisingDiffusion
Policy (D3P), a diffusion policy that dynamically adjusts the
number of denoising steps during task execution. The D3P
architecture consists of two components: a standard noise
predicting network that serves as the base diffusion policy,
and a lightweight adaptor (Fig. 1). The adaptor is designed
to predict thenoise-level stridesbased on the current ob-
servation, which eventually adapts the number of denoising
steps for the current action. We mathematically formulate
the dynamic denoising problem as a two-layer partially ob-
servable Markov decision process (POMDP). Then we use
reinforcement learning (RL) with this two-layer POMDP to
jointly train both the base diffusion policy and the adaptor.
Specifically, a base policy is fine-tuned with DPPO (Ren
et al. 2024) to maximize task success. Concurrently, the
lightweight adaptor is trained from scratch with PPO (Schul-
man et al. 2017), where its reward function incentivizes
minimizing denoising steps without compromising perfor-
mance. A key challenge in this joint training process is to
balance the contradictory objectives of task performance and
inference efficiency. To address this, we introduce a three-
stage training strategy that ensures stable convergence and

```
robust performance.
We evaluate D3P on a range of simulated manipulation
tasks. Using the same amount of training data, D3P achieves
an average 2.2×inference speed-up over baseline methods
while maintaining comparable performance. These results
demonstrate that D3P effectively addresses the trade-off be-
tween performance and efficiency. Furthermore, we deploy
D3P on the physical robot. It achieves a 1.9×acceleration in
inference speed against a normal fixed-step diffusion policy.
Overall, our main contributions are as follows:
```
1. Through an empirical study, we reveal that different ac-
    tions contribute unequally to manipulation tasks. There
    exist some crucial actions significantly influencing task
    completion.
2. Building on this observation, we propose Dynamic
    DenoisingDiffusionPolicy (D3P). Trained via RL, D3P
    features a noise-predicting network as the base diffusion
    policy and an adaptor to dynamically adjust its denoising
    steps during task execution.
3. We conduct experiments in eight simulated manipulation
    tasks, demonstrating that D3P achieves the best perfor-
    mance with an averaged 2.2×speed-up over baselines.
4. We demonstrate that D3P can be successfully deployed
    on a physical robot. It achieves a 1.9×acceleration
    against a standard fixed-step diffusion policy.

## Preliminaries

```
Partially Observable Markov Decision Process
(POMDP) We formulate the robot environment
within the framework of a partially observable Markov
decision Process (POMDP), defined by the tuple
MENV := (SENV,AENV,OENV,Pinit,ENV,PENV,RENV).
Here,SENV, AENV, andOENV are the state, action, and
observation space separately. The process begins with
an initial state s 0 ∼ Pinit,ENV. At each environment
timestept, an agent receives an observationot ∈ OENV,
and takes an actionat ∈ AENV according to a policy
π(at|ot). The environment responds by transitioning to
a new statest+1 ∼ PENV(st+1 | st,at)and providing
a rewardrt = RENV(st,at). The objective in RL is to
learn a policyπθ that maximizes the expected return,
Eπθ[Jt(st,at)] := Eπθ
```
```
hP
T− 1
τ=tγ
```
```
τ−t
ENVrτ|st,at
```
```
i
, where
γENV∈(0,1)is a discount factor andTis the the episode
horizon.
Diffusion Models Denoising diffusion probabilistic mod-
els (DDPMs) (Sohl-Dickstein et al. 2015; Ho, Jain, and
Abbeel 2020) frame sample generation as an iterative de-
noising procedure, reversing a length-N diffusion chain
{xi}Ni=0, wherex^0 is a clean sample,xN∼N( 0 ,I)is pure
Gaussian noise, andNis the number of denoising steps. The
denoising is parameterized by a neural network,εθ(xi,i),
trained to predict the noise component within a noisy sam-
plexiat noise-leveli. At inference, this process starts with
a sample of pure noisexNand progressively refines it over
Nsteps, generating a clean samplex^0.
Despite great performance, inference in DDPMs is slow
because it must execute allN reverse steps. Denoising
```

Figure 2:Visualizing Action Criticality via Perturbed Returns.The plots show the predicted perturbed return from
Dφ(ot,at)at different time of the task. A lower return indicates the action more crucial, as a perturbation is more likely to
lead to task failure.

diffusion implicit models (DDIMs) (Song, Meng, and Er-
mon 2020) accelerate sampling by replacing the stochas-
tic reverse process with a deterministic mapping that allows
larger strides in the noise schedule. Using the same training
loss as DDPMs, DDIMs traverse only a sparse set of noise
levelsτ 0 > τ 1 >···> τS= 0whereS≪N. At a given
leveli, the sampler can skipk > 1 noise-levels and go di-
rectly to a less–noisy pointxi−k:

```
xi−k∼N
```
#### 

```
μ(xi,εi,i,k), η σ^2 iI
```
#### 

#### , (1)

whereσicomes from the predefined schedule. Settingη= 0
removes the stochastic term, making the process fully deter-
ministic and much faster for inference

Diffusion Policies Diffusion policy (DP) (Chi et al.
2023) treats a diffusion model as the control policyπθ.
To preserve temporal consistency, DP predicts an action
chunkXt={at,...,at+Ta− 1 }conditioned on the current
observation ot. With the DDIM sampler, the denoising
update at noise leveliis

```
Xti−^1 ∼N
```
#### 

```
μ
```
#### 

```
Xti,εθ(Xti,ot,i),i,k
```
#### 

```
, η σi^2 I
```
#### 

#### , (2)

withkthe stride. This iterative procedure forms a POMDP
MDN := (SDN,ADN,ODN,Pinit,DN,PDN,RDN). This pro-
cess starts with a state of pure noiseXNt ∼ N( 0 ,I)and
terminates ati= 0to produce the final action chunk.

## Empirical Study: Identifying Crucial Actions

We conduct an empirical study to validate that not all ac-
tions in a task are equally important. An action is deemed
crucial if a disturbance on it significantly degrades task per-
formance. Otherwise, it is considered routine.

Experiment Design We start from a pre-trained expert
policyπexpert that achieves over 90% success in the en-
vironmentMENV. At a randomly selected timestept ∈
{ 0 ,...,T− 1 }we add Gaussian noiseξ to the expert
action at, yielding a perturbed action a′t = at +ξ.
The episode then resumes underπexpert. The perturbation’s
impact is measured by the episode return,J(st,a′t) =

Eπexpert

```
hP
T
τ=0γ
```
```
τ−t
ENVrτ
```
(^) st,a′t
i
.A lowJ signals that the
originalatwas crucial. We fit a lightweight predictorDφ:
OENV×AENV→Rthat predictsJfrom the unperturbed
pair(ot,at). More implementation details of our empirical
study appear inAppendix A.
Key Findings Fig. 2 plotsDφ’s predicted return on the
SquareandTransporttasks from Robomimic (Man-
dlekar et al. 2021). The results demonstrate that the
criticality of actions throughout a task is highly non-
uniform. ForSquare, fine interactions, such as grasping
the block and aligning it with the peg, receive the lowest
predicted returns (5.04, 3.95), marking them as crucial.
Whereas broad arm motions score higher (7.58, 8.56) and
are routine. After completion, the return peaks at 13.
because later actions no longer influence success. For
Transport, grasping the hammer (3.39) and the bimanual
hand-over (3.69) dominate task success, while transit mo-
tions are routine. Overall, crucial actions are concentrated in
direct physical interactions, such as grasping and handover,
while broader movements have a relatively minor impact
on the results. This observation motivates our approach:
adaptively allocating denoising steps to different actions,
focusing more capacity on the crucial actions.

## Method

```
In this section, we introduceDynamicDenoisingDiffusion
Policy (D3P), a method designed to dynamically adjust the
number of denoising steps during task execution. D3P uti-
lizes the noise-level stride scheme in DDIM solver (Song,
Meng, and Ermon 2020), and augments a base diffusion pol-
icyπθwith an adaptorKωtrained to pick strides, as shown
in Fig. 1(c). The adaptor observes
```
#### 

```
ot, Xit
```
#### 

```
and outputs a
stride: large strides skip more noise levels to speed up rou-
tine actions, while small strides keep precise denoising for
crucial actions. We frame adaptive denoising as a two-layer
POMDP and train the base policyπθand adaptorKωjointly
with RL. During the training,πθis fine-tuned to maximize
task reward, whileKωlearns to cut denoising steps without
degrading success. A three-stage training strategy is used
to stabilize training. The full algorithm is summarized in
Alg. 1.
```

Figure 3: We formulate the dynamic denoising problem as a two-layer POMDP, where a denoising process (MDN) is nested
within the environment (MENV). At each step, the adaptorKωpredicts the noise-level strides. The base diffusion policy and
the adaptor are jointly updated via RL.

### Problem Formulation

Following Psenka et al. (2023); Ren et al. (2024), we de-
fine a two-layer POMDPM= (S,A,O,Pinit,P,R)that
nests the denoising processMDNinside the environment
MENV(Fig. 3). In this two-layer PODMPM, we denote
a time index as ̄t(t,i) =tN+ (N−i−1), wheretis
the environment step andi∈[0,N]is the index of noise
level. The states and observations inMare all tuples, de-
noted ass ̄ ̄t= (st,Xti),o ̄ ̄t= (ot,Xti),withst∈ SENVand
Xti∈SDN. An episode starts att ̄(0,N)with ̄st ̄(0,N)∼Pinit,
andPinitis defined as

```
Pinit= (Pinit,ENV, Pinit,DN). (3)
```
At each time step ̄t, the action tuple(a ̄t,k ̄t)is generated by
the base policyπθand the adaptorKωfollowing Eq. (4),
where⌊·⌋indicates rounding down.

```
k ̄t(t,i)∼Kω(k ̄t| ̄o ̄t), j ̄t= max
```
#### 

```
i−⌊k ̄t(t,i)⌋, 0
```
#### 

#### ,

```
a ̄t∼πθ(a ̄t| ̄ot ̄,i) :=N
```
#### 

```
μ(Xit,εθ( ̄ot ̄,i),i,k ̄t),ησ^2 iI
```
#### 

#### .

#### (4)

Eq. (4) indicates thatKωpredicts the noise-level stridesk ̄t
to further control the total denoising steps. Theηis set to 1
during training, making the Gaussian likelihood ofπθcom-
putable, and is set to 0 when inference. After taking actions,
Mmakes transition following Eq. (5).

```
̄s ̄t+1∼
```
#### (

```
(δst,δat ̄) , j > 0 ,

PENV(st+1|st,Xt^0 ),Pinit,DN
```
#### 

```
, j= 0.
```
#### (5)

In Eq. (5),j > 0 signifies an incomplete denoising pro-
cess, and both the environment statestand observastionot
remain unchanged. Whenj= 0, the denoising process con-
cludes and the resulting action is executed in the environ-
ment. Subsequently,MENVtransitions to a new state accord-
ing toPENV, and a new pure noise,XtN+1, is resampled from
Gaussian distribution.

### Dynamic Denoising

Following the problem formulation, we jointly trainπθand
Kωon the same batch of rollout data.

```
To fine-tune the base policy, we apply DPPO (Ren et al.
2024) to optimizeπθand its value criticVΘon the dynamic-
denoising POMDPM. The criticVΘ(ot), conditioned on
the environment observationot, later serves as a proxy for
task performance when training the adaptor.
The adaptor aims to reduce the total denoising steps with-
out degrading performance. The most direct performance
signal is the binary success flagrs, ̄t∈{ 0 , 1 }, yet its sparsity
and high variance impede stable learning. Instead, we em-
ploy the advantage of each state–action pair inMENVas a
dense, low-variance, but slightly biased metric, computed as
AˆΘ(Xt^0 ,ot) =Jπθ(st,Xt^0 )−VΘ(ot), (6)
whereJπθ(st,Xt^0 )is the discounted return defined in the
Preliminaries. A largerAˆΘindicates that the fully denoised
actionX^0 tis beneficial for future returns.
To encourage shorter denoising chains, we introduce a
discount factorγs∈(0,1)that penalizes longer denoising
process. Balancing task performance against computational
cost, we define the reward for adaptor as
```
```
rK, ̄t(t,i)=
```
#### 

#### 

#### 

```
0 , j > 0
```
```
αAˆΘ
```
#### 

```
Xt^0 ,ot
```
#### 

```
γssgnt×stpt
```
```
+β rs, ̄tγsstpt
```
```
, j= 0
```
#### (7)

```
whereαandβare weighting coefficients, sgntis the sign of
the advantageAˆΘ(Xt^0 ,ot), and stptdenotes the total denois-
ing steps for generating the clean actionXt^0. This formula-
tion rewards advantageous actions and successful episodes
while exponentially penalizing long denoising sequences.
We updateKωwith PPO (Schulman et al. 2017), mini-
mizing the clipped PPO loss in Eq. (8).
```
```
−LCLIP(ω) =−EˆKωold
```
```
h
min
```
#### 

```
ρ(ω)Aˆωold(kt ̄,o ̄ ̄t),
```
```
clip(ρ(ω), 1 −εclip,1 +εclip)Aˆωold(k ̄t,o ̄ ̄t)
```
```
i
.
```
#### (8)

```
Here,ρ(ω) = KKωω(kt ̄|o ̄ ̄t)
old(k ̄t|o ̄t ̄)
```
```
is the probability ratio between
```
```
the current and old adaptor. The advantageAˆωoldis computed
using Generalized Advantage Estimation (GAE) (Schulman
et al. 2015) based on the rewardrK, ̄t.
```

Algorithm 1: Dynamic Denoising Diffusion Policy (D3P)

1:Input:Pretrained base policyπθ, dynamic denoising environmentM, discount factorsγENV,γs, environment horizonT,
max denoise stepsN, warm-up denoising stepsc, update epochse,eslow, stage thresholdζ 1 ,ζ 2.
2:Initialize:AdaptorKωwith mean ofc, empty bufferB.
3:whileAveragers,bart< ζ 1 do ▷Stage 1: Warm-up
4: Warm-upπθusing DPPO (Ren et al. 2024) with fixed denoising stepsc
5:whilenot convergeddo ▷Stage 2: Joint training
6: Reset environment to ̄t(0,N)as Eq. (3)
7: whilet < Tdo ▷Rollout data
8: Sample action(k ̄t,X ̄ ̄t)following Eq. (4). GetlogK ̄t,logπ ̄t
9: Execute the actions. Perform transition as Eq. (5). Get ̄o ̄t+1andrπ, ̄tfollowing (Ren et al. 2024).
10: Add

#### 

```
o ̄ ̄t,(k ̄t,a ̄t),(logKω(k ̄t),logπθ(a ̄t)),rπ, ̄t
```
#### 

```
to bufferB.
```
11: Calculate the advantageAˆΘusing Eq. (6). Get success flagrs, ̄trefer to task results. ▷Prepare for update

12: SetrK, ̄tas Eq. (7), usingAˆΘand rs, ̄t.
13: forepoch= 0, 1 ,...,e− 1 do ▷Policy optimization
14: Update parameterθandΘusing DPPO (Ren et al. 2024)
15: Update parameterωwith PPO loss in Eq. (8)

16: ifAverage stp< ζ 2 thene←eslow ▷Stage 3: Conservative fine-tuning

17: Return:Trained base policyπθand adaptorKω.

### Training Strategy

Directly training the base policy and adaptor from scratch
often causes instability and even collapse. To address this,
D3P adopts a three-stage training strategy preceded by
behavior cloning of the base policyπθ on pre-collected
datasets.

Stage 1: Base DP Warm-up. We first warm upπθwith
DPPO (Ren et al. 2024) while keeping the denoising steps
fixed atc. The warming up proceeds until the task success
rate exceeds a preset thresholdζ 1. This stage can be skipped
if a sufficiently strong base policy is already available.

Stage 2: Joint Training. Next, we initialize the adaptor
Kωas a Gaussian policy with mean ofcand variance ofv^2.
Then we trainKωandπθjointly. Each iteration collects a
batch of trajectories followed byePPO update epochs for
both modules.

Stage 3: Conservative Fine-tuning. When the average
denoising steps satisfiesE[stpt]< ζ 2 , we switch to a conser-
vative stage. This stage mirrors Stage 2 but uses fewer up-
date epochs per iteration. This precaution prevents the adap-
tor from shrinking the denoising steps so aggressively that it
destabilizes the base policy and leads to a collapse.
The full algorithm of D3P, including the three-stage
schedule, is summarized in Alg. 1.

## Experiments

To evaluate D3P, we conduct comprehensive robot manip-
ulation experiments in both simulation and the real world.
We first detail the experimental setups, then benchmark D3P
against several baselines. Subsequently, we present ablation
studies to analyze the contribution of each component in our
framework. Finally, we demonstrate the deployment of D3P
onto a physical robot, highlighting its practical applicability.

### Setups

```
Environments We evaluate our method on eight manipu-
lation tasks from two benchmarks: Robomimic (Mandlekar
et al. 2021) and Franka Kitchen (Gupta et al. 2019). These
environments include simple pick-and-place tasks and chal-
lenging long-horizon, multi-stage assembly tasks.
```
```
Baselines We compare D3P against three representative
baselines covering different paradigms: (1) DPPO (Ren
et al. 2024), a state-of-the-art (SOTA) algorithm foronline
fine-tuningDPs, (2) consistency policy (CP) (Prasad et al.
2024), adistillation-based accelerationmethod, and (3) Fal-
con (Chen et al. 2025), a training-freestreamingapproach.
To ensure a fair comparison, all policies are pre-trained on
the same dataset. Using the pre-trained policy, we train D3P
and the DPPO policy with an identical amount of online RL
data. The Falcon and consistency policy baselines are sub-
sequently derived from the DPPO fine-tuned policy.
```
```
Metrics We evaluate task performance using success rate
and episodic return (J =
```
#### PT

```
τ=0rτ) where higher values
indicate better performance. To assess computational effi-
ciency, we follow Prasad et al. (2024); Chen et al. (2025)
and adopt the Number of Function Evaluations (NFE) per
action as our primary metric. NFE provides a fair compari-
son of inference cost because all methods use the same base
network architecture. For DPPO and CP, the NFE count is
equivalent to the number of denoising steps. For Falcon, its
selection mechanism requires an additional base policy in-
ference, adding one NFE per action. For our method, we
only count the NFE from the base policy, as our lightweight
adaptor has fewer than 1 / 15 the parameters of the base pol-
icy. All results are averaged over three random seeds, with
evaluations conducted on 100 episodes per seed.
For additional details on the experimental setup, please
refer toAppendix B.
```

```
(a) Lift (State) (b) Can (State) (c) Square (State) (d) Square (Pixel)
```
```
(e) Transport (State) (f) Transport (Pixel) (g) Kitchen-complete-v0 (h) Kitchen-mixed-v
```
Figure 4: We plot success rate / episodic return against the Number of Function Evaluations (NFE) per action. We use episodic
return for the simplerLiftandCantasks because success rates for most methods saturate above 90%. D3P achieves the best
in performance and efficiency across all tasks. D3P matches or surpasses the peak performance of the 10-step DPPO baseline
while achieving an average 2.2×speed-up. All results are averaged over 3 seeds with 100 evaluation episodes per seed.

### Performance Evaluation

Fig. 4 compares our D3P against the baselines by plotting
their task performance versus inference cost. On the simpler
LiftandCantasks, the success rates for most methods sat-
urate above 90% and we use episodic return as a more fine-
grained performance metric. In this figure, an ideal method
would occupy the upper-left corner, signifying high perfor-
mance achieved at a low inference cost.
As expected, the performance of the DPPO fine-tuned
policy correlates with its inference cost. Reducing the NFEs
improves training and inference speed, yet leads to a clear
performance drop in success rate and return. Similarly, while
Falcon accelerates the 10-step DPPO policy, its performance
consistently decreases as the acceleration ratio increases.
The success rate (or return) of Falcon degrades sharply un-
der 6 NFE. Consistency Policy, through distillation, enables
few-step or even single-step inference. However, the distil-
lation process creates a performance ceiling that prevents it
from matching the optimal, multi-step teacher policy.
In contrast, D3P dynamically adapts its denoising effort,
using more denoising steps only when necessary. This al-
lows D3P to match or even exceed the peak performance
of the 10-step DPPO policy while achieving an average in-
ference speed-up of 2.2 times. The results unequivocally
demonstrate that D3P establishes a better Pareto frontier,
consistently achieving an optimal performance-efficiency
trade-off than all baselines across all tasks. Due to space

```
constraints, detailed training curves are provided inAp-
pendix C.
```
### Ablation study

```
We perform ablation studies on the Square and
Kitchen-complete-v0tasks to isolate the contrinbu-
tions of D3P’s key design choices.
Fig. 5 validates the importance of our three-stage train-
ing strategy. Removing stage 1 leads to a significant perfor-
mance drop at the start of the training, while removing stage
3 causes unstable curves during later training. Our full three-
stage approach effectively warms up the policy and then sta-
bilizes the fine-tuning process, proving crucial for guiding a
robust, optimal convergence.
Fig. 6 shows the analysis of our reward formulation in
Eq. (7). Settingα = 0leaves only the unbiased but de-
layed success reward. This high-variance signal makes train-
ing unstable and prone to failure. Settingβ= 0leaves only
the low-variance but biased advantage term. While the ad-
vantage term stabilizes training, the policy is more likely to
converge to a suboptimal solution.
```
### Real-world Deployment

```
To demonstrate D3P’s effectiveness in the physical world,
we deploy the policy on a Franka robot arm. All inference
was performed on a consumer-grade desktop (i7-12900K
CPU, RTX 2080 GPU). We mitigate the visual sim-to-real
```

```
(a) Square (b) Kitchen-complete-v
```
Figure 5: Ablation of the three-stage training strategy. Re-
moving Stage 1 impairs initial learning, while removing
Stage 3 destabilizes final convergence. The full strategy is
critical for achieving rapid and stable performance.

```
(a) Square (b) Kitchen-complete-v
```
Figure 6: Ablation of our reward formulation. Relying solely
on the success reward (α= 0) causes training instability,
while using only the advantage term (β= 0) leads to subop-
timal convergence. Both components are essential for stable
training towards an optimal policy.

gap with a latent diffusion model (Rombach et al. 2021) that
aligns real-world images simulated ones. As illustrated in
Fig. 13, D3P successfully performs theSquaretask. For
crucial actions such as grasping and aligning, D3P increases
its denoising steps to 8 and 6, to generate accurate actions.
Conversely, for simpler motions, it reduces the step count
to as low as 3. D3P achieves the control frequency of 33.
Hz, a 1.92×speedup over the 17.59 Hz of a fixed 10-step
diffusion policy. Additional deployment details are provided
inAppendix D.

## Related Work

Optimizing Diffusion Policies via RL To overcome the
data dependency of imitation learning (Chi et al. 2023;
Pearce et al. 2023; Wang et al. 2024b; Prasad et al.
2024), many methods use RL to optimize DPs. In offline
RL, methods adapt DPs using techniques derived from Q-
learning (Wang, Hunt, and Zhou 2022; Hansen-Estruch et al.
2023) or policy gradients (Kang et al. 2023). In the on-

```
Figure 7: Real-world demonstration of D3P performing the
Squaretask. The plot shows that D3P dynamically adjusts
the number of denoising steps during task execution. It al-
locates more steps for crucial actions, such as grasping and
aligning, and fewer for routine movements.
```
```
line setting, DPs are often trained within actor-critic frame-
works (Wang et al. 2024a; Yang et al. 2023; Ren et al. 2024;
Li et al. 2024) or with action-gradients from a learned Q-
function (Psenka et al. 2023). However, these methods apply
a fixed number of denoising steps for all actions, leaving the
critical issue of slow inference speed unresolved.
```
```
Accelerating Diffusion Policy Inference To improve the
inference speed of DPs, a straightforward way is reduce the
number of denoising steps. Prior work primarily use pol-
icy distillation (Prasad et al. 2024; Wang et al. 2024b) or
streaming denoising (Høeg, Du, and Egeland 2024; Chen
et al. 2025). A key limitation is that these methods treat all
actions as equally important. In contrast, our method, D3P,
adaptively adjusts the computational effort for each action.
While the concept of adaptive denoising exists for single-
image generation (Ye et al. 2025), the sequential decision-
making of robotics presents distinct challenges with long-
term rewards and temporal dependencies. We address this by
formulating the dynamic denoising problem as a two-layer
POMDP and jointly optimizing the base diffusion policy and
the adaptor. The training process is stabilized by a special-
ized reward and a three-stage training strategy.
```
## Conclusion

```
In this work, we introduced Dynamic Denoising Diffu-
sion Policy (D3P), a diffusion policy capitalizing on the
varying action criticalities in robotic tasks. D3P employs a
lightweight adaptor to dynamically adjust denoising steps,
assigning more steps to crucial actions and fewer to rou-
tine ones. We use RL to joint optimize the base policy and
the adaptor with a carefully-designed reward and a three-
stage training strategy. Our simulation experiments demon-
strate that D3P achieves an averaged 2.2×inference speed-
up over baselines without compromising task success. Fur-
thermore, D3P is deployed on a physical robot, achieving
a 1.9×inference acceleration against a fixed-step diffusion
policy. These results underscore the potential of adaptive in-
ference in robot learning, developing more efficient policies
for real-time applications.
```

## References

Chen, H.; Liu, M.; Ma, C.; Ma, X.; Ma, Z.; Wu, H.;
Chen, Y.; Zhong, Y.; Wang, M.; Li, Q.; and Yang, Y. 2025.
Falcon: Fast Visuomotor Policies via Partial Denoising.
arXiv:2503.00339.

Chi, C.; Xu, Z.; Feng, S.; Cousineau, E.; Du, Y.; Burchfiel,
B.; Tedrake, R.; and Song, S. 2023. Diffusion policy: Visuo-
motor policy learning via action diffusion.The International
Journal of Robotics Research, 02783649241273668.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.;
Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y.

2020. Generative adversarial networks.Communications of
the ACM, 63(11): 139–144.

Gupta, A.; Kumar, V.; Lynch, C.; Levine, S.; and Hausman,
K. 2019. Relay policy learning: Solving long-horizon tasks
via imitation and reinforcement learning. arXiv preprint
arXiv:1910.11956.

Hansen-Estruch, P.; Kostrikov, I.; Janner, M.; Kuba, J. G.;
and Levine, S. 2023. Idql: Implicit q-learning as an
actor-critic method with diffusion policies. arXiv preprint
arXiv:2304.10573.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion
probabilistic models. Advances in neural information pro-
cessing systems, 33: 6840–6851.

Høeg, S. H.; Du, Y.; and Egeland, O. 2024. Streaming Diffu-
sion Policy: Fast Policy Synthesis with Variable Noise Dif-
fusion Models.arXiv preprint arXiv:2406.04806.

Janner, M.; Du, Y.; Tenenbaum, J.; and Levine, S. 2022.
Planning with Diffusion for Flexible Behavior Synthesis.
In Chaudhuri, K.; Jegelka, S.; Song, L.; Szepesvari, C.;
Niu, G.; and Sabato, S., eds.,Proceedings of the 39th In-
ternational Conference on Machine Learning, volume 162
ofProceedings of Machine Learning Research, 9902–9915.
PMLR.

Kang, B.; Ma, X.; Du, C.; Pang, T.; and Yan, S. 2023. Ef-
ficient diffusion policies for offline reinforcement learning.
Advances in Neural Information Processing Systems, 36:
67195–67212.

Kingma, D. P.; Welling, M.; et al. 2013. Auto-encoding vari-
ational bayes.

Li, S.; Krohn, R.; Chen, T.; Ajay, A.; Agrawal, P.; and
Chalvatzaki, G. 2024. Learning multimodal behaviors from
scratch with diffusion policy gradient.Advances in Neural
Information Processing Systems, 37: 38456–38479.

Loshchilov, I.; and Hutter, F. 2016. Sgdr: Stochas-
tic gradient descent with warm restarts. arXiv preprint
arXiv:1608.03983.

Lu, C.; Zhou, Y.; Bao, F.; Chen, J.; Li, C.; and Zhu, J. 2022a.
Dpm-solver: A fast ode solver for diffusion probabilistic
model sampling in around 10 steps. Advances in Neural
Information Processing Systems, 35: 5775–5787.

Lu, C.; Zhou, Y.; Bao, F.; Chen, J.; Li, C.; and Zhu, J. 2022b.
Dpm-solver++: Fast solver for guided sampling of diffusion
probabilistic models.arXiv preprint arXiv:2211.01095.

```
Ma, X.; Patidar, S.; Haughton, I.; and James, S. 2024. Hi-
erarchical diffusion policy for kinematics-aware multi-task
robotic manipulation. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
18081–18090.
Mandlekar, A.; Xu, D.; Wong, J.; Nasiriany, S.; Wang, C.;
Kulkarni, R.; Fei-Fei, L.; Savarese, S.; Zhu, Y.; and Mart ́ın-
Mart ́ın, R. 2021. What Matters in Learning from Offline
Human Demonstrations for Robot Manipulation. InarXiv
preprint arXiv:2108.03298.
Pearce, T.; Rashid, T.; Kanervisto, A.; Bignell, D.; Sun, M.;
Georgescu, R.; Macua, S. V.; Tan, S. Z.; Momennejad, I.;
Hofmann, K.; and Devlin, S. 2023. Imitating Human Be-
haviour with Diffusion Models. arXiv:2301.10677.
Prasad, A.; Lin, K.; Wu, J.; Zhou, L.; and Bohg, J. 2024.
Consistency policy: Accelerated visuomotor policies via
consistency distillation.arXiv preprint arXiv:2405.07503.
Psenka, M.; Escontrela, A.; Abbeel, P.; and Ma, Y. 2023.
Learning a diffusion model policy from rewards via q-score
matching.arXiv preprint arXiv:2312.11752.
Ren, A. Z.; Lidard, J.; Ankile, L. L.; Simeonov, A.; Agrawal,
P.; Majumdar, A.; Burchfiel, B.; Dai, H.; and Simchowitz,
M. 2024. Diffusion policy policy optimization. arXiv
preprint arXiv:2409.00588.
Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Om-
mer, B. 2021. High-Resolution Image Synthesis with Latent
Diffusion Models. arXiv:2112.10752.
Schulman, J.; Moritz, P.; Levine, S.; Jordan, M.; and
Abbeel, P. 2015. High-dimensional continuous control
using generalized advantage estimation. arXiv preprint
arXiv:1506.02438.
Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and
Klimov, O. 2017. Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347.
Shafiullah, N. M.; Cui, Z.; Altanzaya, A. A.; and Pinto, L.
```
2022. Behavior transformers: Cloningkmodes with one
stone.Advances in neural information processing systems,
35: 22955–22968.
Sohl-Dickstein, J.; Weiss, E.; Maheswaranathan, N.; and
Ganguli, S. 2015. Deep unsupervised learning using
nonequilibrium thermodynamics. InInternational confer-
ence on machine learning, 2256–2265. pmlr.
Song, J.; Meng, C.; and Ermon, S. 2020. Denoising diffusion
implicit models.arXiv preprint arXiv:2010.02502.
Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Er-
mon, S.; and Poole, B. 2020. Score-based generative model-
ing through stochastic differential equations.arXiv preprint
arXiv:2011.13456.
Wang, Y.; Wang, L.; Jiang, Y.; Zou, W.; Liu, T.; Song, X.;
Wang, W.; Xiao, L.; Wu, J.; Duan, J.; and Li, S. E. 2024a.
Diffusion Actor-Critic with Entropy Regulator. In Glober-
son, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tom-
czak, J.; and Zhang, C., eds.,Advances in Neural Informa-
tion Processing Systems, volume 37, 54183–54204. Curran
Associates, Inc.


Wang, Z.; Hunt, J. J.; and Zhou, M. 2022. Diffusion poli-
cies as an expressive policy class for offline reinforcement
learning.arXiv preprint arXiv:2208.06193.

Wang, Z.; Li, Z.; Mandlekar, A.; Xu, Z.; Fan, J.; Narang, Y.;
Fan, L.; Zhu, Y.; Balaji, Y.; Zhou, M.; et al. 2024b. One-
step diffusion policy: Fast visuomotor policies via diffusion
distillation.arXiv preprint arXiv:2410.21257.

Yang, L.; Huang, Z.; Lei, F.; Zhong, Y.; Yang, Y.; Fang, C.;
Wen, S.; Zhou, B.; and Lin, Z. 2023. Policy representation
via diffusion probability model for reinforcement learning.
arXiv preprint arXiv:2305.13122.

Ye, Z.; Chen, Z.; Li, T.; Huang, Z.; Luo, W.; and Qi, G.-
J. 2025. Schedule on the fly: Diffusion time prediction
for faster and better image generation. InProceedings of
the Computer Vision and Pattern Recognition Conference,
23412–23422.

Ze, Y.; Zhang, G.; Zhang, K.; Hu, C.; Wang, M.; and Xu,
H. 2024. 3d diffusion policy: Generalizable visuomotor pol-
icy learning via simple 3d representations. arXiv preprint
arXiv:2403.03954.

Zhao, T. Z.; Kumar, V.; Levine, S.; and Finn, C. 2023. Learn-
ing fine-grained bimanual manipulation with low-cost hard-
ware.arXiv preprint arXiv:2304.13705.


## A Implementation Details of Empirical Study

Our empirical study reveals that not all actions in a robotic task contribute equally to the results. This section details the specifics
of our empirical study.
As introduced in theEmpirical Study, we train a return predictor, denoted asDφ:OENV×AENV→R, to predict the
subsequent return based on current observationotand the unperturbed actionat. This predictor is trained via supervised
learning on a dataset collected from Monte-Carlo rollouts. The detailed procedure for data collection and training is presented
in Alg. 2, with all notations consistent with the main text.

Algorithm 2: Train a Return PredictorDφ

```
1:Input:Task environmentMENV, expert policyπexpert, number of episodesL, discount factorγENV, episode horizonT,
update intervallint, update epochEp, max buffer lengthM, variancev.
2:Initialize:NetworkDφ, Empty bufferBwith max lengthM.
3:forl= 0, 1 ,...,L− 1 do
4: Reset environment to geto 0
5: Sampletl∼Uniform{ 0 , 1 ,...,T− 1 }
6: fort= 0, 1 ,...,T− 1 do
7: Get expert actionat∼πexpert(·|ot)
8: ift==tlthen
9: ξ∼N
```
#### 

```
0 ,v^2
```
#### 

,a′t←at+ξ
10: Executea′tto getot+1andrt
11: else
12: Executeatto getot+1andrt

13: CalculateJl←

#### PT

```
τ=0γ
```
τ−tl
ENVrτ
14: Add(otl,atl,Jl)toB
15: ifl%lint== 0then
16: fore= 0, 1 ,...,Ep− 1 do
17: UpdateDφonBby minimizing the loss:L(φ) =L^1

#### P

```
(o,a,J)∈B(Dφ(o,a)−J)
```
```
2
```
18: Return:Trained predictorDφ.

We parameterizeDφas a 5-layer MLP with hidden layer sizes of[256, 512 , 1024 , 512 ,256]. For our experiments on the
SquareandTransporttasks, we useL= 600andL= 350, respectively. The full training hyperparameters are provided
in Tab. 1, with training curves shown in Fig. 8. Detailed settings for both tasks are available inAppendix B.

```
(a) Square (b) Transport
```
```
Figure 8: Training loss of the return predictorDφversus training episodes for the (a)Squareand (b)Transporttasks
```

```
Hyperparameter Value
max buffer sizeM 100000
number of parallel environments 10
discount factorγENV 0.
noise variancev 0.
update intervallint 20
update epochEp 6
learning rate 0.
weight decay 0.
```
```
Table 1: Hyperparameters for training the return predictorDφ.
```
```
(a) Lift (b) Can (c) Square (d) Transport
```
```
Figure 9: Four manipulation tasks in Robomimic
```
## B Additional details of simulation experiments

### B.1 Environment and Dataset

Environment We evaluate D3P in two benchmarks:

1. Robomimic(Mandlekar et al. 2021). We evaluate our method on manipulation tasks from the Robomimic suite, including
    the simple pick-and-place tasksLiftandCan, the assembly taskSquare, and the bimanual handover taskTransport.
    Fig. 9 demonstrate the four manipulation tasks.
2. Franka Kitchen(Gupta et al. 2019). We also use the Franka Kitchen environment, a benchmark for challenging long-
    horizon, multi-stage manipulation. As illustrated in Fig. 10, the robot need to complete 4 subtasks in sequence: open the
    microwave, move the kettle, flip the light switch, and slide open the cabinet door. We use two settings in this environ-
    mrnt: (1)Kitchen-complete-v0, where the policy is pretrained on a dataset of successful demonstrations, and (2)
    Kitchen-mixed-v0, where the pretraining dataset contains contains various subtasks being performed, but the 4 target
    subtasks are never completed in sequence together.

Figure 10: Franka Kitchen environment. In the environment, the robot need to complete 4 subtasks in sequence: open the
microwave, move the kettle, flip the light switch, and slide open the cabinet door.


```
Benchmark Task Obs dim (State) Obs dim (Pixel) Act dim×Ta T Sparse reward
```
```
Franka Kitchen
```
```
kitchen-complete-v0 60 - 9 × 4 280 Yes
kitchen-mixed-v0 60 - 9 × 4 280 Yes
```
```
Robomimic
```
```
Lift 19 - 7 × 4 300 Yes
Can 23 - 7 × 4 300 Yes
Square 23 - 7 × 4 400 Yes
Transport 59 - 14 × 8 800 Yes
Square (Pixel) 9 (3, 96 ,96)× 1 7 × 4 400 Yes
Transport (Pixel) 18 (3, 96 ,96)× 2 14 × 8 800 Yes
```
Table 2: We detail the task configurations in this table, where“Obs dim (State)” is the dimension of the state observation, “Obs
dim (Image)” is the dimension of the pixel observation, “Act dim” is the dimension of a single action,Tais the action chunck
horizon, andTis the episode horizon.

We list the task configurations in Tab. 2. In this table, “Obs dim (State)” indicates the dimension of the state observation,
“Obs dim (Image)” is the dimension of the pixel observation, Act dim×Tashows the shape of an action chunk, andTis the
episode horizon.
All tasks employ a sparse reward. In Robomimic tasks, the agent receives a reward of 0 for every timestep prior to task
completion. Upon successful completion, the agent is awarded a +1 reward for all subsequent timesteps until the episode
concludes. Therefore, the episodic return directly reflects the agent’s quality. A higher return indicates faster task completion,
whereas a lower return suggests the task was only finished near the end of the episode. If the task is not completed within the
maximum episode length ofTsteps, the total episodic return is 0. For the Franka Kitchen environment, the agent receives a +
reward upon the completion of each sub-task. As there are 4 sub-tasks, the maximum possible episodic return is 4.

Dataset We pre-train the policies on the dataset provided by one of our baseline, DPPO (Ren et al. 2024). As acknowledged
by the authors of DPPO the dataset includes suboptimal data.

### B.2 Baseline

We compare D3P against three representative baselines that cover different paradigms for improving or acclerating diffusion
policies.

DPPO Diffusion policy policy optimization (DPPO) (Ren et al. 2024) is a state-of-the-art algorithm foronline fine-tuning
diffusion policies that operate with a fixed number of denoising steps. DPPO models the problem as a two-layer POMDP
to leverage the sequential nature of the diffusion denoising process. It directly optimizes the entire denoising chain using an
actor-critic framework. At each iteration, the diffusion policyπθis updated with the following PPO-style loss function:

```
Lθ=E
```
#### 

```
min
```
#### 

```
Aˆπθold( ̄o ̄t,a ̄t)πθ( ̄o ̄t,a ̄t)
πθold( ̄o ̄t,a ̄t)
```
```
,Aˆπθold( ̄o ̄t,at ̄)clip
```
#### 

```
πθ( ̄o ̄t,a ̄t)
πθold( ̄o ̄t,a ̄t)
```
```
, 1 −εclip,1 +εclip
```
#### 

#### , (9)

where ̄t(t,i) =tN+ (N−i−1)is the time index in the two-layer POMDP,o ̄ ̄tis the observation, anda ̄tis the action. The

advantageAˆπθis estimated as shown in Eq. (10), using a discount factor ofγDENOISEi.

```
Aˆπθ( ̄o ̄t,a ̄t) =γiDENOISE(Jπθ( ̄o ̄t,a ̄t)−VΘ( ̄o ̄t)). (10)
```
Notably, DPPO employs a dynamic clipping parameterεclip, whose value is determined by the denoising progress,t= 1−Ni,
as defined in Eq. (11).

```
εclip=εbase+ (εcoef−εbase)·
```
```
eεratet− 1
eεrate− 1
```
#### . (11)

Here,εbase,εcoef, andεrateare hyperparameters. DPPO provides structured online exploration and enhances training stability. In
experiments, we use the official hyperparameter settings from the original implementation, detailed in Tab. 3.

Consistency Policy A Consistency Policy (CP) (Prasad et al. 2024) is created by distilling a pretrained diffusion policy, which
enforces self-consistency along the teacher model’s learned trajectories. This distillation enables CP to sample actions using
very few denoising steps. We denote the CP asgθ(ot,Xti,i,s), which is conditioned on the current observationot, the current
action chunkXit, the current noise leveli, and a target noise levels < i. The distillation loss combines a Denoising Score
Matching (DSM) loss and a Consistency Trajectory Model (CTM) loss:

```
LDSM=Ex 0 , i[d(x 0 ,gθ(ot,xi,i,0))],
LCTM=Ex 0 ,i,s
```
#### 

```
d
```
#### 

```
gθ
```
#### 

```
ot,gθ(o,Xti,i,s),s, 0
```
#### 

```
, gθ
```
#### 

```
ot,gθ(o,Xti−^1 ,i− 1 ,s),s, 0
```
#### 

#### .

#### (12)


```
Hyperparameter Value
Number of parallel environments 40
Condition horizonTo 1
Reward discount factorγENV 0.
Advantage discount factorγDENOISE 0.
GAEλ 0.
Optimizer AdamW
Actor learning rate 1e-
Actor weight decay 0
Critic learning rate 1e-
Critic weight decay 0
Batch size 10000
Value loss coefficient 0.
Entropy loss coefficient 0
Clip coefficient baseεbase 0.
Clip coefficientεcoef 0.
Clip coefficient rateεrate 3
Max gradient norm 10.
Rollout steps 400
Update epoch 10
Denoisingη 1.
```
```
Table 3: Hyperparameters of DPPO
```
In these equations,Xit−^1 is generated fromXtiby the teacher policy, andd(x,y)is a pseudo-Huber loss function measuring
the distance betweenxandy, as shown in Eq. (13)

```
d(x,y) =
```
```
q
∥x−y∥^22 +c^2 −c. (13)
```
During training, the final loss is a weighted sum ofLDSMandLCTMwith coefficientswDSMandwCTM, respectively. For
our experiments, we distill CPs from our DPPO-finetuned policies. The hyperparameters for distillation are listed in Tab. 4.

Falcon Falcon (Chen et al. 2025) is a training-freestreamingmethod that accelerates diffusion policies through partial de-
noising. The core insight of Falcon is to leverage the sequential dependency inherent in such tasks. Instead of initiating the
denoising process from a standard Gaussian distribution for every action, Falcon reuses a partially denoised action from a latent
buffer of historical actions, thereby significantly reducing the required number of sampling steps.
The selection of this action prior is managed by a thresholding mechanism. Falcon first uses the unexecuted action sequence
from the previous timestep as a reference. Then, for each partially denoised action aXτiin the latent buffer, Falcon computes

a one-step estimationXˆt^0 conditioned on the current observationotwith Tweedie’s formula (Eq. (14)), whereεθis the noise-

```
Hyperparameter Value
Weight of DSM losswDSM 1.
Weight of CTM losswCTM 1.
Optimizer AdamW
Batch size 512
Training epoch 1500
Learning rate 1e-
Weight decay 1e-
```
```
Learning rate scheduler
```
```
CosineAnnealingWarmupRestart
(Loshchilov and Hutter 2016)
First cycle steps 1500
Warmup steps 100
Minimum learning rate 1e-
```
```
Table 4: Hyperparameters of consistency policy
```

predicting network defined in Preliminaries, andα ̄iis a set of parameters determinated by a fixed noise schedule.

```
Xˆt^0 =E[Xˆ^0 t|ot,Xit,i] =X
```
```
t
t−
```
#### √

```
1 −α ̄iεθ(ot,Xti,i)
√
α ̄i
```
#### . (14)

Then we form a candidate setS, consists of actions whose one-step estimations are within a certain distanceεFalconof the
reference action. The final prior actionXitis then sampled from this set with a probability that favors lower noise levels,
modulated by a temperature parameterκ. This thresholding mechanism allows Falcon to find the best action priors.

### B.3 Hyperparameters of D3P

The D3P training hyperparameters are provided in Tabs. 5 and 6. The settings for the base policy are based on those from
DPPO. In contrast, for the adaptor, we keep most hyperparameters unchanged but specifically tune several key parameters, such
asζ 1 ,ζ 2 and the reward weightβ.

```
Fine-tuning Base policyπθ Training AdaptorKω
Hyperparameter Value Hyperparameter Value
Number of parallel environments 40 Condition horizonTo 1
Condition horizonTo 1 Denoising step discountγs 0.
Reward discount factorγENV 0.999 reward discount factorγ 0.
Advantage discount factorγDENOISE 0.99 GAEλ 0.
GAEλ 0.95 Optimizer AdamW
Optimizer AdamW Adaptor weight decay 1e-
Actor learning rate 1e-4 Batch size 40000
Actor weight decay 0 Value loss coefficient 1.
Critic learning rate 1e-3 Entropy loss coefficient 0.
Critic weight decay 0 Clip coefficient 0.
Batch size 10000 Max gradient norm 10.
Value loss coefficient 0.5 Update epoch 10
Entropy loss coefficient 0 Rollout steps 400
Clip coefficient baseεbase 0.001 Reward weightα 1.
Clip coefficientεcoef 0.
Clip coefficient rateεrate 3
Max gradient norm 10.
Rollout steps 400
Denoisingη 1.
```
```
Table 5: Shared hyperparameters of D3P
```
```
Hyperparameter Lift Can
```
```
Square
(State & Pixel)
```
```
Transport
(State & Pixel) kitchen-complete-v0 kitchen-mixed-v
Update epoch 10 10 10 6 10 10
Adaptor learning rate 1e-4 1e-4 1e-4 3e-5 1e-4 1e-
Thresholdζ 1 100.0 170.0 210.0 310.0 3.3 3.
Thresholdζ 2 4.0 4.0 5.0 7.5 5.0 5.
Reward weightβ 0.2 0.2 0.06 0.1 0.4 0.
```
```
Table 6: Task-specific hyperparameters of D3P
```
### B.4 Computation

All experiments are conducted on our server cluster running Ubuntu 22.04. The training environment is built on Python 3.8, with
specific dependencies listed in Listing 1. The hardware resources utilized for training are detailed in Tab. 7. As the Robomimic
and Franka Kitchen simulation environments are primarily CPU-intensive, our training process consumes significant CPU and
memory resources. Each training run is performed using a single GPU.


```
Environment CPU Memory GPU
```
```
Training
```
```
Robomimic (State) 45 Core 128G RTX 3090 24G
Robomimic (Pixel) 45 Core 256G A800 40G
Franka Kitchen 10 Core 64G RTX 3090 24G
Evaluation All environment 16 Core 32G RTX 3090 24G
```
```
Table 7: Computing resources for training D3P
```
Listing 1: Dependencies for training D3P
1 av==12.3.
2 einops==0.8.
3 gdown==5.2.
4 gym==0.22.
5 hydra-core==1.3.
6 imageio==2.35.
7 matplotlib==3.7.
8 omegaconf==2.3.
9 pretty_errors==1.2.
10 torch==2.4.
11 tqdm==4.66.
12 wandb==0.17.
13
14 # for robomimic environment
15 cython<
16 d4rl
17 patchelf
18 mujoco==3.1.
19 robomimic
20 robosuite @ v1.4.
21
22 # for franka kitchen
23 cython<
24 d4rl
25 dm_control==1.0.
26 mujoco==3.1.
27 patchelf


## C Additional Experiment Results

### C.1 Performance Comparison

Fig. 11 presents the success rate and return versus Number of Function Evaluations (NFE) for all methods. In these plots, the
upper-left corner represents the ideal trade-off: high performance with low inference cost. With an equivalent training budget,
D3P consistently matches or surpasses the fixed 10-step diffusion policy baseline across both metrics.
The performance of the DPPO baseline correlates directly with its inference cost, dropping significantly as NFE is reduced.
The two acceleration baselines, CP and Falcon, exhibit distinct features. CP excels at few-step inference, outperforming Falcon
on tasks likeLift,Can, andSquare(State). In contrast, Falcon, a training-free accelerator, achieves a better performance-
efficiency trade-off on more complex tasks such asSquare(Pixel) andTransport(Pixel).
We present training curves in Fig. 12, and summarize quantitative results in Tab. 8. Across eight tasks, D3P achieves an
average success rate of 0.917, nearly matching the 0.918 of the 10-step DPPO baseline. Meanwhile, D3P achieves a 2.2×mean
speed-up. We calculate this per-task acceleration ratio,racc, using Eq. (15).

```
racc=
```
```
Eall episodes
```
```
hP
T
t=0stpDPPO,t
```
```
i
```
```
Eall episodes
```
```
hP
T
t=0stpD3P,t
```
```
i (15)
```
```
Task NFE SRDPPO Return NFE FalconSR Return NFE SRCP Return NFE SRD3P (Ours)Return Acc. Ratio
```
```
Lift
(State)
```
```
10.0 1.00±0.00 160.1±4.4 10.0±0.0 0.96±0.00 154.6±4.2 5.00 0.89±0.01 84.9±2.
4.00 1.003.00 0.99±±0.000.01 126.7120.1±±2.72.7 8.256.41±±0.25 0.960.21 0.94±±0.010.01 158.4145.9±±3.55.9 3.00 0.951.00 1.00±±0.010.00 143.389.3±±2.82.4 3.90±0.12 1.00±0.01 182.4±23.2 2.
```
```
3.60±0.44 0.76±0.03 121.8±9.
```
```
Can
(State)
```
```
10.0 0.98±0.00 204.4±3.4 10.0±0.0 0.97±0.01 194.0±5.8 5.00 0.98±0.01 186.8±3.
4.00 0.943.00 0.89±±0.010.01 187.9173.3±±3.53.7 7.875.22±±0.12 0.900.35 0.80±±0.020.03 183.3117.1±±5.86.4 3.00 0.941.00 0.94±±0.010.00 162.6167.7±±3.33.0 3.71±0.37 0.98±0.02 201.1±8.0 2.
```
```
4.23±0.28 0.67±0.03 96.3±8.
```
```
Square
(State)
```
```
10.0 0.99±0.00 303.1±2.2 10.0±0.0 0.98±0.00 300.4±6.3 5.0 0.88±0.01 244.3±8.
5.00 0.954.00 0.90±±0.03 285.90.04 265.8±±10.012.3 5.414.41±±0.22 0.940.30 0.80±±0.040.04 256.7207.4±±7.38.8 3.01.0 0.880.86±±0.020.01 244.6234.0±±2.62.4 4.05±0.20 0.99±0.01 308.7±5.5 2.
```
```
3.00 0.88±0.04 221.2±14.9 3.03±0.34 0.74±0.06 164.1±8.
Transport
(State)
```
```
10.0 0.80±0.05 323.6±24.3 10.0±0.0 0.76±0.04 321.4±28.4 5.00 0.71±0.06 243.5±24.
7.00 0.58±0.04 233.4±25.0 7.75±0.50 0.76±0.04 300.4±31.1 3.00 0.74±0.03 270.0±19.9 6.75±0.14 0.80±0.07 341.0±30.8 1.
5.00 0.58±0.05 221.2±22.3 6.69±0.89 0.56±0.06 239.6±31.5 1.00 0.64±0.03 235.0±22.
```
```
Square
(Image)
```
```
10.0 0.86±0.05 240.5±14.2 10.0±0.0 0.87±0.04 228.2±14.9 5.00 0.63±0.06 153.1±7.
5.00 0.78±0.07 187.0±20.9 6.894.82±±0.39 0.870.51 0.70±±0.06 212.80.06 162.8±±21.428.1 3.00 0.651.00 0.60±±0.030.03 161.8150.3±±9.64.8 3.95±0.74 0.89±0.03 232.0±3.2 2.
```
```
3.39±0.36 0.64±0.07 128.9±8.
Transport
(Image)
```
```
10.0 0.75±0.02 289.9±13.5 10.0±0.0 0.75±0.02 284.6±9.7 5.00 0.39±0.07 104.8±8.
7.00 0.63±0.03 240.9±12.9 8.12±0.16 0.74±0.04 271.3±10.2 3.00 0.36±0.08 98.4±9.3 6.89±0.52 0.75±0.03 285.0±9.8 1.
6.55±0.71 0.59±0.09 213.9±16.1 1.00 0.45±0.04 119.3±7.
```
```
Kitchen
(Complete)
```
```
10.0 0.98±0.00 3.96±0.02 10.0±0.0 0.99±0.01 3.98±0.03 5.0 0.76±0.03 3.48±0.
5.00 0.944.00 0.90±±0.010.01 3.913.83±±0.030.03 7.925.52±±0.18 0.980.39 0.82±±0.010.05 3.953.56±±0.070.08 3.01.0 0.670.65±±0.010.01 3.172.99±±0.020.02 4.15±0.37 0.98±0.01 3.98±0.02 2.
```
```
4.54±0.31 0.77±0.05 3.50±0.
```
```
Kitchen
(Mixed)
```
```
10.0 0.99±0.00 3.99±0.00 10.0±0.0 0.99±0.01 3.98±0.05 5.0 0.85±0.06 3.72±0.
5.004.00 0.00.0±±0.00.0 2.992.99±±0.010.00 7.795.76±±0.15 0.990.21 0.90±±0.010.05 3.883.44±±0.040.09 3.01.0 0.810.90±±0.060.05 3.573.80±±0.090.07 4.78±0.02 0.97±0.02 3.93±0.05 2.
```
```
4.40±0.37 0.70±0.07 3.22±0.
```
```
Table 8: Detailed results of all methods, fomulated as mean±std. All results are averaged over 5 seeds.
```

```
(a) Lift (State) - Success Rate (b) Can (State) - Success Rate (c) Square (State) - Success Rate (d) Square (Pixel) - Success Rate
```
```
(e) Transport (State)
```
- Success Rate

```
(f) Transport (Pixel)
```
- Success Rate

```
(g) Kitchen-complete-v
```
- Success Rate

```
(h) Kitchen-mixed-v
```
- Success Rate

```
(i) Lift (State) - Return (j) Can (State) - Return (k) Square (State) - Return (l) Square (Pixel) - Return
```
```
(m) Transport (State)
```
- Return

```
(n) Transport (Pixel)
```
- Return

```
(o) Kitchen-complete-v
```
- Return

```
(p) Kitchen-mixed-v
```
- Return

Figure 11: We plot success rate and episodic return against the NFE. An ideal method occupies the upper-left corner, represent-
ing high performance at a low inference cost. D3P matches or surpasses the peak performance of the 10-step DPPO baseline
while achieving an average 2.2×speed-up. All results are averaged over 5 seeds.


(a) Lift (State) - Success Rate (b) Can (State) - Success Rate (c) Square (State) - Success Rate (d) Square (Pixel) - Success Rate

```
(e) Transport (State)
```
- Success Rate

```
(f) Transport (Pixel)
```
- Success Rate

```
(g) Kitchen-complete-v
```
- Success Rate

```
(h) Kitchen-mixed-v
```
- Success Rate

```
(i) Lift (State) - Return (j) Can (State) - Return (k) Square (State) - Return (l) Square (Pixel) - Return
```
```
(m) Transport (State)
```
- Return

```
(n) Transport (Pixel)
```
- Return

```
(o) Kitchen-complete-v
```
- Return

```
(p) Kitchen-mixed-v
```
- Success Rate

```
Figure 12: Training curves of eight tasks. Each curve is averaged over 5 random seeds.
```

## D Real-world Deployment

We deploy D3P on a Franka robot arm to complete theSquaretask.

Setup We define the task environment as illustrated in Fig. 13a. In theSquaretask, the agent is required to successfully
grasp the handle of the square nut and precisely mate it with the corresponding square peg. We use a Franka robot arm for
executing, and a consumer-grade desktop (i7-12900K CPU, RTX 2080 GPU) for computing. We fabricated the physical objects
required for the task using 3D printing, ensuring that their color, shape, and dimensions are consistent with simulated objects,
as shown in Fig. 13b. For this task, D3P utilizes images and joint positions as input. The RGB images are captured by an Intel
RealSense D435i camera positioned 0.92m in front of and 0.53m above the robot’s base, providing a 45 degree downward
viewing angle. Fig. 14 shows the progress of this task, and we include a video in the supplementary materials that showcases
the agent executing this task.

```
(a)SqaureTask Setting (b) We use 3D print to manufacture the objects
```
```
Figure 13: TheSquaretask involves: (1) grasping the handle of the square, (2) inserting the square onto corresponding peg.
```
```
Figure 14: Demonstration of theSquaretask
```
Sim-to-real transfer To achieve sim-to-real transfer without real-world data, we employ a latent diffusion model
(LDM) (Rombach et al. 2021) to convert real-world images into a style that approximates the simulated domain, as illus-
trated in Fig. 15. The process begins with a pretrained Variational Autoencoder (VAE) that encodes the input image into a
compact latent feature. Subsequently, a diffusion model operates on this latent feature to perform the translation. Finally, the
VAE’s decoder reconstructs the processed feature into the converted output image.
We train the LDM with paires of images collected in simulation by domain radomization. As illustrated in Fig. 16, we employ
domain randomization in the simulation by varying the object materials and lighting conditions, from which we collect paired
images of the canonical and randomized scenes.
Finally, to bridge the gap in camera parameters between the simulated and real-world environments, we introduce a curricu-
lum learning strategy during RL training. This curriculum systematically adjusts the camera’s intrinsic and extrinsic parameters,
progressively transitioning them from the initial simulation settings to the physical hardware setting.


```
(a) Real-world Image (b) Converted Image
```
Figure 15: We use LDM to convert real-world images into the simulation style

```
(a) Canonical Image (b) Randomized Image
```
```
Figure 16: Images pair for training LDM
```

