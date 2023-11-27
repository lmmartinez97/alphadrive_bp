# Introduction
This repository contains the code associated to the PhD thesis _Behavior Planning in Autonomous Driving: Application of Deep Reinforcement Learning Techniques in Critical Maneuvers.

## Folder structure.
This section will be updated as the project is developed.

As of 11/15/2023:
```bash
├── envs
│   └── preprocessing.env
│   └── carla.env
├── highD
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
│       └── saved_models
├── simulator
│   ├── agents
│   │   ├── __init__.py
│   │   ├── navigation
│   │   │   ├── basic_agent.py
│   │   │   ├── behavior_agent.py
│   │   │   ├── behavior_types.py
│   │   │   ├── constant_velocity_agent.py
│   │   │   ├── controller.py
│   │   │   ├── global_route_planner.py
│   │   │   ├── __init__.py
│   │   │   ├── local_planner.py
│   │   └── tools
│   │       ├── __init__.py
│   │       ├── misc.py
│   ├── carla_client.py
│   ├── modules
│   │   ├── camera.py
│   │   ├── hud.py
│   │   ├── __init__.py
│   │   ├── keyboard_control.py
│   │   ├── logger.py
│   │   ├── printers.py
│   │   ├── sensors.py
│   │   └── shared_mem.py

└── ui
    └── README_ui.md

```

## Software requirements
Python environments can be found within the envs directory. Any environment will be used to execute every file under the directory it shares its name with - i.e. the environment "preprocessing" can execute every notebook and script under /scene_representation/preprocessing.

Some parts of the project are designed to be executed inside of a docker container. The instructions to execute such files are not ready yet.



