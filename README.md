CS 294-112 | Deep Reinforcement Learning Fall 2018 - Assignment Solutions
===============

The course website: http://rail.eecs.berkeley.edu/deeprlcourse/

My own solutions for Cs294-112
<br><br>

## Project1
  <br>***Behavioral Cloning vs DAgger*** <br><br>
  I was able to get the results below with given hyperparameter.

  <img src="./hw1/images/hw1_performance.png" width = "500" style = "margin-left:50px">

  <br>**Learning Curves** <br><br>
   *Hopper-v2* <br><br>
  <img src="./hw1/images/Hopper-v2.png" width = "700" style = "margin-left:50px">
  <br><br>*Reacher-v2* <br><br>
  <img src="./hw1/images/Reacher-v2.png" width = "700" style = "margin-left:50px">

  <br>Agents with huge improvements in DAgger have shown soaring loss function in learning curves.
<br><br><br><br>

## Project2
  <br>***Policy Gradient Method in discrete action space and continous action space*** <br><br>

  *FrozenLake-v2*<br>
  <img src="./hw2/plot/CartPole-v0/lb_(rtg)_dna.png" width = "400" style = "margin-left:50px">
  <img src="./hw2/plot/CartPole-v0/lb_rtg_(dna).png" width = "400" style = "margin-left:50px">

  <br>*HalfCheetah-v2*<br>
  <img src="./hw2/plot/HalfCheetah-v2/batch50000_baseline.png" width = "400" style = "margin-left:50px">

  1. Reward-to-go has shown improvements in performance
  2. Normalizing the advantageous function has shown reduction of the high variance
  3. Providing baseline has shown reduction of the high variance
