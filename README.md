# Artificial Intelligence projects

This repository holds 4 different AI projects made within the context of the Artificial Intelligence course taught at ULB in the 3rd year of the Computer Science bachelor degree. This course is heavily based on the CS188 course given at the UC Berkeley. 

Each part implements a different theme in the AI field:

- **Planning and research**
- **Probabilistic reasoning** (Bayesian Networks)
- **Decision making with uncertainty** (Reinforcement Learning)
- **Machine learning** (Neural Networks)

The details about each subject can be found in the README file of each subdirectory (and the scientific reports).

## Repository structure

```
.
├── general-images/ (directory containing images used in the reports, I've centralized them to avoid duplication)
├── planning-and-research/
│   ├── doc/
│   │   └── ...
│   ├── src/
│   │   └── ...
│   ├── tests/
│   │   └── ...
│   └── ...
├── probabilistic-reasoning/
│   ├── doc/
│   │   └── ...
│   └── src/
│       └── ...
├── decision-making-with-uncertainty/
│   ├── doc/
│   │   └── ...
│   └── src/
│       └── ...
└── machine-learning/
    ├── doc/
    │   └── ...
    ├── src/
    │   └── ...
    └── data/
        └── ...
```

Each part contains a ```src``` directory that contains the code implementing the AI method. The ```doc``` directories contain the directives for each part given by the staff of the course. They also contain a detailed analysis of the results in form of a scientific report (in french).

## Running the projects

Each part has different requirements. In order to run them, you must initialize a python virtual environment by typing:
```bash
python3 -m venv .venv
```
(considering you already have Python3 installed)

Then you activate it:
```bash
source .venv/bin/activate
```

Finally, install the requirements:
```bash
pip install -r requirements.txt
```

If you are wondering why I didn't make a ```requirements.txt``` for each project, let me be honest. It was pure laziness.