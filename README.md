# Introduction

This repository contains the code associated to the PhD thesis _Behavior Planning in Autonomous Driving: Application of Deep Reinforcement Learning Techniques in Critical Maneuvers.

## Folder structure

This section will be updated as the project is developed.

As of 16/01/2024:

```bash
├── envs
│   ├── carla.env
│   └── preprocessing.env
├── pseudocode.md
├── pseudocode.pdf
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
│           ├── autoencoder_1_configs.json
│           ├── autoencoder_1_decoder.h5
│           ├── autoencoder_1_encoder.h5
│           ├── autoencoder_1.h5
│           ├── autoencoder_test_configs.json
│           ├── autoencoder_test.h5
│           ├── autoencoder_test_history1_configs.json
│           ├── autoencoder_test_history1.h5
│           ├── autoencoder_test_history_configs.json
│           ├── autoencoder_test_history.h5
│           ├── autoencoder_test_int_1_configs.json
│           ├── autoencoder_test_int_1.h5
│           ├── autoencoder_test_int_3_configs.json
│           ├── autoencoder_test_int_3.h5
│           └── autoencoder_test_int_3_notes.txt
├── simulator
│   ├── agents
│   │   ├── navigation
│   │   │   ├── basic_agent.py
│   │   │   ├── behavior_agent.py
│   │   │   ├── behavior_types.py
│   │   │   ├── constant_velocity_agent.py
│   │   │   ├── controller.py
│   │   │   ├── global_route_planner.py
│   │   │   └── local_planner.py
│   │   └── tools
│   │       └── misc.py
│   ├── carla_client.py
│   ├── modules
│   │   ├── camera.py
│   │   ├── hud.py
│   │   ├── keyboard_control.py
│   │   ├── logger.py
│   │   ├── potential_field.py
│   │   ├── printers.py
│   │   ├── sensors.py
│   │   ├── shared_mem.py
│   │   ├── state_manager.py
│   │   └── utils.py
│   ├── pseudocode.py
│   └── server.py
├── tests
│   ├── async_logging
│   │   ├── client.py
│   │   ├── server_dash.py
│   │   ├── server.py
│   │   └── server_qt.py
│   └── socket_logging
│       ├── client
│       │   ├── client_logger.py
│       │   └── client_socket.py
│       ├── server
│       │   ├── server.py
│       │   └── server_socket.py
│       └── web_app
│           ├── app.py
│           └── templates
└── ui
    └── README_ui.md
```

## Software requirements

Python environments can be found within the envs directory. Any environment will be used to execute every file under the directory it shares its name with - i.e. the environment "preprocessing" can execute every notebook and script under /scene_representation/preprocessing.

Some parts of the project are designed to be executed inside of a docker container. The instructions to execute such files are not ready yet.
