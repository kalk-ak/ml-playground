# Argubots

This repository contains a collection of "argubots" - dialogue agents that can engage in debates on a variety of topics. The project is designed to be a platform for developing and evaluating different strategies for argumentation and persuasion.

## Features

*   **A variety of argubots:** The repository includes a range of argubots with different strategies, from simple rule-based bots to more sophisticated LLM-based agents.
*   **Kialo integration:** The argubots can use data from Kialo.com, a platform for structured debates, to inform their arguments.
*   **Evaluation framework:** The project includes a framework for evaluating the performance of the argubots.
*   **Extensible:** The project is designed to be easily extensible, so you can add your own argubots and evaluation methods.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/argubots.git
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the Kialo data:**
    The Kialo data is included in the `data` directory.

## Usage

To evaluate the argubots, run the `run_evaluation.py` script:
```bash
python3 src/run_evaluation.py
```
This will run a series of simulated dialogues between the argubots and the characters defined in `src/characters.py`. The results of the evaluation will be printed to the console.

You can create your own argubots by subclassing the `Agent` class in `src/agents.py`. You can also create your own characters by adding to the `characters.py` file.

## Project Structure

```
├── data/
│   ├── ... (Kialo discussion files)
│   └── LICENSE
├── src/
│   ├── agents.py           # Base classes for agents
│   ├── argubots.py         # The argubots
│   ├── characters.py       # The characters to debate against
│   ├── dialogue.py         # Classes for representing dialogues
│   ├── evaluate.py         # The evaluation framework
│   ├── kialo.py            # Kialo data processing
│   ├── run_evaluation.py   # The main script for running evaluations
│   └── ...
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

**Note on the data:** The Kialo discussion data in the `data` directory is for private use within this project, as per the permission from Kialo. If you wish to use the data for other purposes, please contact Kialo at support@kialo.com.
