# Introduction

This repository contains the code associated to the PhD thesis _Behavior Planning in Autonomous Driving: Application of Deep Reinforcement Learning Techniques in Critical Maneuvers.

## Folder structure

This section will be updated as the project is developed.

As of 31/01/2024:

```bash
├── README.md
├── envs
│   ├── carla.env
│   └── preprocessing.env
├── misc
│   ├── pseudocode_review
│   │   ├── pseudocode.md
│   │   ├── pseudocode.pdf
│   │   └── pseudocode.py
│   └── tests
│       ├── async_logging
│       │   ├── client.py
│       │   ├── server.py
│       │   ├── server_dash.py
│       │   └── server_qt.py
│       ├── server.py
│       └── socket_logging
│           ├── client
│           │   ├── client_logger.py
│           │   └── client_socket.py
│           ├── server
│           │   ├── server.py
│           │   └── server_socket.py
│           └── web_app
│               ├── app.py
│               └── templates
├── scene_representation
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── field_extraction.ipynb
│   │   ├── field_extraction.py
│   │   ├── group_extraction.ipynb
│   │   ├── group_extraction.py
│   │   └── read_csv.py
│   └── training
│       ├── autoencoder_training.ipynb
│       └── saved_models
│           ├── ...
├── simulator
│   ├── __init__.py
│   ├── alphazero.py
│   ├── carla_client.py
│   ├── checkpoints
│   └── modules
│       ├── __init__.py
│       ├── agents
│       │   ├── __init__.py
│       │   ├── navigation
│       │   │   ├── __init__.py
│       │   │   ├── basic_agent.py
│       │   │   ├── behavior_agent.py
│       │   │   ├── behavior_types.py
│       │   │   ├── constant_velocity_agent.py
│       │   │   ├── controller.py
│       │   │   ├── global_route_planner.py
│       │   │   └── local_planner.py
│       │   └── tools
│       │       ├── __init__.py
│       │       └── misc.py
│       ├── camera.py
│       ├── hud.py
│       ├── keyboard_control.py
│       ├── logger.py
│       ├── mcts
│       │   ├── __init__.py
│       │   ├── helpers.py
│       │   ├── network.py
│       │   ├── self_play.py
│       │   └── utils.py
│       ├── potential_field.py
│       ├── printers.py
│       ├── sensors.py
│       ├── shared_mem.py
│       ├── simulation.py
│       ├── state_manager.py
│       ├── utils.py
│       └── world.py
└── ui
    └── README_ui.md
```

## Software requirements

Python environments can be found within the envs directory. Any environment will be used to execute every file under the directory it shares its name with - i.e. the environment "preprocessing" can execute every notebook and script under /scene_representation/preprocessing.

Some parts of the project are designed to be executed inside of a docker container. The instructions to execute such files are not ready yet.
