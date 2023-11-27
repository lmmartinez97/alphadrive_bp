# Introduction
This repository contains the code associated to the PhD thesis _Behavior Planning in Autonomous Driving: Application of Deep Reinforcement Learning Techniques in Critical Maneuvers.

## Folder structure.
This section will be updated as the project is developed.

As of 11/15/2023:
```bash
scene_representation
├── preprocessing
│   ├── group_extraction.ipynb
│   ├── group_extraction.py
│   ├── field_extraction.ipynb
│   ├── field_extraction.py
│   ├── read_csv.py
└── training
    ├── autoencoder_training.ipynb
    └── saved_models

ui
└── README_ui.md -- EMPTY

envs
└── preprocessing.env
```

## Software requirements
Python environments can be found within the envs directory. Any environment will be used to execute every file under the directory it shares its name with - i.e. the environment "preprocessing" can execute every notebook and script under /scene_representation/preprocessing.

Some parts of the project are designed to be executed inside of a docker container. The instructions to execute such files are not ready yet.



