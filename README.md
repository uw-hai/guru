# Guru

## Description

A reinforcement learning agent that maximizes crowdsourcing quality by teaching and testing.

References:

[1] Jonathan Bragg, Mausam, and Daniel S. Weld. 2016. [Optimal Testing for Crowd Workers](https://www.cs.washington.edu/ai/pubs/bragg-aamas16.pdf). In Proceedings of the 15th International Conference on Autonomous Agents and Multiagent Systems (AAMAS '16). Singapore.

[2] Jonathan Bragg, Mausam, and Daniel S. Weld. 2015. [Learning on the Job: Optimal Instruction for Crowdsourcing](https://www.cs.washington.edu/ai/pubs/bragg-icml15.pdf). In ICML '15 Workshop on Crowdsourcing and Machine Learning. Lille, France.

## Installation

We recommend installing your environment using Miniconda. Once Miniconda is installed, you can a create virtual environment with the correct dependencies by running
```conda env create -f environment.yml python=2.7 -n $ENV```
where `$ENV` is the name of your virtual environment.

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

If you want to use the main experiment or visualization code, you will need to set up a MongoDB database.
The application loads the configuration settings for the database from environment variables. An example:
```bash
export APP_SETTINGS=viz_app_config.Config
export MONGO_HOST=127.0.0.1
export MONGO_PORT=27017
export MONGO_USER=your_username
export MONGO_PASS=your_password
export MONGO_DBNAME=your_dbname
export MONGO_AUTH_DBNAME=your_authentication_dbname
```

## Running

The main end-to-end experiment code lives in `exp.py`.

If you would like to use the agent in your own application, you may want to follow the usage in `exp.py` as a guideline for how to initialize the agent from `policy.py`. More detailed instructions to come.

TODO: Describe visualization application.

## Testing

Use the provided `./test.sh` script.

## Contact

Please create an issue, pull request, or send email to jbragg [at] cs.washington.edu.
