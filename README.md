# Project Title

Benchmark Project to run Machine Learning, Deep Learning and GenAI Workloads



## Installation

### Prerequisites

List any software or tools needed to run the project. For example:

- Python (3.11)
- Docker
- Micromamba

### Setup

Instructions on how to set up the project:

1. Clone the repository:
    ```
    git clone https://github.com/HenrikMader/Benchmark.git
    ```

2. Install dependencies (for example, using `pip` for Python):
    On x86:
    ```
    pip install -r requirements.txt
    ```
    On POWER:
    ```
    micromamba install -c rocketce -c defaults pytorch matplotlib flask==2.0.3 Werkzeug==2.0.3 scikit-learn 'conda-forge::gradio'

    Micromaba install -c conda-forge cvxopt

    pip install pm4py

    (if necessary install other packages through micromamba and rocketce)
    ```


3. Run machine learning and deep learning python script

    naviagate to either machine learning or deep learning folder and execute:

    python main.py

4. Run GenAI Workloads

    On x86:
    - CPU:
        - docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
        - docker exec -it ollama ollama run granite3-dense:8b
    - GPU:
        - docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
        - docker exec -it ollama ollama run granite3-dense:8b
    
    On POWER:
        - docker run -d --privileged -v ollama:/root/.ollama -p 11434:11434 --name ollama quay.io/mgiessing/ollama:v0.3.14
        - docker exec -it ollama ollama run granite3-dense:8b
