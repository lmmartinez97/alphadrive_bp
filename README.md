# Introduction

This repository contains the code associated to the Behavior Plannign branch of the AlphaDrive project. 

The bulk of this project is being developed under a PhD thesis by Luis Miguel Martinez: _Behavior Planning in Autonomous Driving: Application of Deep Reinforcement Learning Techniques in Critical Maneuvers_.

## Folder structure

This section will be updated as the project is developed.

As of 23/04/2024:

```bash
├── envs
│   ├── carla.env
│   └── preprocessing.env
├── misc
│   ├── pseudocode_review
│   │   ├── pseudocode.md
│   │   ├── pseudocode.pdf
│   │   └── pseudocode.py
│   └── tests
│       ├── async_logging ...
│       ├── server.py ... 
│       └── socket_logging ...
├── README.md
├── scene_representation
│   ├── preprocessing
│   │   ├── field_extraction.ipynb
│   │   ├── field_extraction.py
│   │   ├── group_extraction.ipynb
│   │   ├── group_extraction.py
│   │   ├── __init__.py
│   │   └── read_csv.py
│   └── training
│       ├── autoencoder_training.ipynb
│       └── saved_models ...
├── simulator
│   ├── alphazero.py
│   ├── carla_client.py
│   ├── __init__.py
│   └── modules
│       ├── agents
│       │   ├── __init__.py
│       │   ├── navigation
│       │   │   ├── basic_agent.py
│       │   │   ├── behavior_agent.py
│       │   │   ├── behavior_types.py
│       │   │   ├── constant_velocity_agent.py
│       │   │   ├── controller.py
│       │   │   ├── global_route_planner.py
│       │   │   ├── __init__.py
│       │   │   └── local_planner.py
│       │   └── tools
│       │       ├── __init__.py
│       │       └── misc.py
│       ├── carla
│       │   ├── autoencoder.py
│       │   ├── camera.py
│       │   ├── hud.py
│       │   ├── __init__.py
│       │   ├── keyboard_control.py
│       │   ├── logger.py
│       │   ├── mpc.py
│       │   ├── pid.py
│       │   ├── potential_field.py
│       │   ├── printers.py
│       │   ├── sensors.py
│       │   ├── shared_mem.py
│       │   ├── simulation.py
│       │   ├── state_manager.py
│       │   ├── utils.py
│       │   └── world.py
│       ├── __init__.py
│       └── mcts
│           ├── helpers.py
│           ├── __init__.py
│           ├── network.py
│           ├── self_play.py
│           └── utils.py
└── ui
    └── README_ui.md

```

## Software requirements

Python environments can be found within the envs directory. Any environment will be used to execute every file under the directory it shares its name with - i.e. the environment "preprocessing" can execute every notebook and script under /scene_representation/preprocessing.

Some parts of the project are designed to be executed inside of a docker container. The instructions to execute such files are not ready yet.
