# Action Trading for Self-Interested Multi-Agent Reinforcement Learning in a Smart Factory Setting

![action_trading](https://github.com/Anemosx/action-trading/blob/trading/smart_factory_trading.png?raw=true)

## Abstract

Cooperation is essential for all living beings, as it is a key component to success. Since
multi agent systems have become more and more important in our daily lives, the demand
for cooperating agents has also increased. Agents in multi agent environments try to
maximize their reward as they interact with the environment, which mostly results rather
in selfish than cooperative behaviour. Recent work on cooperation between entities in
multi agent environments have introduced the concept of trading markets, which enables
agents to interact with each other and thus enable cooperation. These trading markets
enable agents to make an offer, consisting of an action that the other agent can follow.
As the actions of the agents are extended with offers, they were able to outperform
agents without this possibility. While cooperation between agents not only led to a
higher overall reward, individual rewards also increased. These results have shown that
the cooperation between agents with extended actions can have a positive impact on
the reward and we therefore take a closer look at the cooperation emerging from the
extended actions. Further investigating the behaviour of agents, we compare the shortand
long-term cooperation with a modified version of action trading and a new trading
mode. While the agents in the original action trading were only able to trade one action
for reward, we want to scale this approach to 𝑛 offer actions. We therefore split the trade
into two separate parts, offer and supply. As we increase the amount of offer actions we
will be able to examine the behaviour and reward of short- and long-term cooperation. In
addition, we will analyze the cooperative behavior of agents as we change various factors
that influence the trade, such as the payment timing, the amount of compensation or
the trading budget. For this purpose we use reinforcement learning and deep q-networks
to train the agents to maximize their rewards in the environment smart factory. In the
smart factory the agents have to compete with each other in order to process machines for
rewards. This competition encourage the agents to trade with each other and therefore
create cooperation. Our results show that the adaptation of action trading also works
in the environment smart factory. The new introduced trading mode reveals that agents
rather cooperate short- than long-term, which is also reflected in the rewards. While
the increase of compensation shows mixed results, the budgeting of the trade only has
negative impact on the reward. Also the variation in the payment timing shows that
agents are more likely to cooperate if they pay afterwards.

[Presentation](https://github.com/Anemosx/action-trading/blob/trading/action_trading_pres.pdf)

[Full Thesis](https://github.com/Anemosx/action-trading/blob/trading/action_trading.pdf)

![action_trading_plot](https://github.com/Anemosx/action-trading/blob/trading/smart_factory_trading_plot.png?raw=true)
