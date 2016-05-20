# crowd-teach-rl

## Description

A reinforcement learning agent that maximizes crowdsourcing quality by teaching and testing.

References:

[1] Jonathan Bragg, Mausam, and Daniel S. Weld. 2016. [Optimal Testing for Crowd Workers](https://www.cs.washington.edu/ai/pubs/bragg-aamas16.pdf). In Proceedings of the 15th International Conference on Autonomous Agents and Multiagent Systems (AAMAS '16). Singapore.

[2] Jonathan Bragg, Mausam, and Daniel S. Weld. 2015. [Learning on the Job: Optimal Instruction for Crowdsourcing](https://www.cs.washington.edu/ai/pubs/bragg-icml15.pdf). In ICML '15 Workshop on Crowdsourcing and Machine Learning. Lille, France.

## Requirements

Requires python 2.7.

## Installation

In order to use policies that utilize POMDPs, you must install a supported POMDP solver. The recommended solver is [ZMDP](https://github.com/trey0/zmdp). Once you have installed and built this solver, be sure that you can run `pomdpsol-zmdp` from the shell by either aliasing or adding the ZMDP binary to your `$PATH` environment variable. One way to do this is
```bash
mkdir ~/.bin
cd ~/.bin
ln -s ~/.bin/pomdpsol-zmdp $ZMDP_BIN_DIR/zmdp
```
where the ZMDP binary directory is `$ZMDP_BIN_DIR`. Then in your `~/.bashrc` file, add the line
```bash
export PATH=$PATH:~/.bin
```

## Running

TODO: Describe.

## Testing

TODO: Describe.

## Contact

Please create an issue, pull request, or send email to jbragg [at] cs.washington.edu.
