## About

This project explores how reinforcement learning (RL) trading agents can learn to manipulate financial markets through collusion. The repository contains code and resources related to this investigation.

## Contents

- `agents.py`: Defines the RL trading agents used in the simulations.
- `config.py`: Configuration settings for the simulations.
- `simulate.py`: Main script for running individual simulations.
- `simulate_batch.py`: Script for running batch simulations.
- `util.py`: Utility functions supporting the project.
- `jobs/`: Directory containing job scripts for running simulations on HPC clusters.
  - `batch_sim/`: Contains PBS scripts for batch simulations.
- `papers/`: Directory containing relevant research papers.
- `archive/`: Archived notebooks, experimental files, and generated plots.
- `__pycache__/`: Directory containing Python cache files.

## Requirements

- Python 3.8+
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/moyuanchen/collusion.git
cd collusion

2. (Optional) Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To run a single simulation:
```bash
python simulate.py
```

For batch simulations:
```bash
python simulate_batch.py --sigma_u 0.1
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. Ensure your changes adhere to the established code style and include tests if applicable.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Special thanks to the contributors and the community for their support.
- Inspired by research in financial markets and reinforcement learning.

## Contact

For issues or inquiries, please contact the maintainer (moyuan.chen24 [guess what this is] imperial.ac.uk) or open an issue on GitHub.