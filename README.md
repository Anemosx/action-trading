# Combining cooperative Game theory and MARL
       
Whereas agents in cooperative environments share the same reward function in self-interested settings agents maximize their individual returns. As a consequence agents learned policies tend to be overly greedy especially when shared resources between agents are scarce. Cooperative game theory provides a theoretical framework for self-interested settings where binding agreements between agents are possible. In this work we propose binding agreements for multi-agent reinforcement learning which allow agents to transfer reward in exchange for following a non greedy trajectory. Thus, agents are enabled to compensate each other for their occurred losses that arise from behaving non-greedy. We evaluate our proposed method in a smart-factory scenario where agents compete for available machines. The tasks either have low-priority or high-priority, giving small rewards or high-rewards respectively. We empirically demonstrate that RL agents stably learn agreements and thus originate higher returns compared with the pure self-interested case.

 
![alt text](https://raw.githubusercontent.com/kyrillschmid/contracting-agents/tree/master/experiments/20190923-10-58-52/rewards-var.png)

 
 TODO: 
 
 - Add Action Trading Baseline
 - Add A2C Baseline
 - n > 2?
 
 
 
 
 - DDQN, Duelling Network
 

 