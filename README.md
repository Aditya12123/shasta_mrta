# SHaSTA (Simulator for Human and Swarm Team Applications)

![Simulator Banner](docs/images/overview.png)
*(Screenshot for illustration purposes only.)*

Welcome to the SHaSTA! An open-source simulation platform to study human-swarm applications. The platform provides a unified interface for humans and learning algorithms to command/learn swarm robot behaviors. SHaSTA also provides a unique ability to collect physiological information from humans and time synchronize it with swarm simulation.


## Table of Contents

- [SHaSTA (Simulator for Human and Swarm Team Applications)](#shasta-simulator-for-human-and-swarm-team-applications)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Code of Conduct](#code-of-conduct)
  - [License](#license)

## Features

- Realistic swarm simulation powered by the pybullet.
- Human-Swarm Interaction interface.
- Physiological signal (EEG, EMG, Eye movement, etc.) collection facility built on lab streaming layer.
- OpenAI Gym interface for reinforcement learning experiments.
- Highly customizable simulation environment.
- Easy-to-use and well-documented codebase.

## Installation

1. Clone this repository to your local machine using:

   ```
   git clone https://github.com/ub-nsf-hsi/shasta-ub.git

   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Configure your XPlane installation and simulator settings as needed.

## Usage

1. Launch the simulator by running:

   ```
   python main.py
   ```
   If you want to run a particular block of code, replace the `skip` by `run`.

2. Explore the simulation environment, adjust settings, and start flying!


## Contributing

We welcome contributions from the community to improve and expand this simulator project. If you're interested in contributing, please follow our [Contribution Guidelines](CONTRIBUTING.md).

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) when participating in our community. We strive to maintain a respectful and inclusive environment for everyone.

## License

This project is licensed under the [MIT License](LICENSE), which means you are free to use, modify, and distribute the code in your projects.

---

We hope you enjoy using the SHaSTA. Happy flying!

