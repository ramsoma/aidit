# AI Diagnosis Training (AIDiT)

*Pronounced "Aid-it"*

Training open source models to generate better clinical diagnoses. 

## Problem
Currently LLMS treat health related queries as a question - answer type situation. This leads to a scenario where the LLM provides a bunch of information to the user but not a clear diagnosis or next steps to follow. Prior research has showed that this leads to suboptimal outcomes in both a) improper diagnosis since the member might not formulate the question with enough information initially b) being less actionable or more confusing to the user ([study][https://arxiv.org/pdf/2504.18919]). 

This repo asks the the question - can we post-train LLMs to respond like a doctor would ? Not jump to an answer straight away but collect all information and make a diagnosis only when enough information is collected? 

## Approach 

We start with the base Qwen3 model and post-train it with SFT and DPO to create a LLM that behave much more like a diagnostician. At each step, the thinking tokens maintain the differential diagnosis based on all the available information. Based on this, the LLM picks the next most important question to ask the member. Finally, we teach the LLM some good manners to ask the member if they are ok to continue answering more questions so that this does not feel like a bottomless pit of questions.

We start with some realistic conversation data between a doctor and a patient from this [paper](https://www.nature.com/articles/s41597-022-01423-1). We create a training set by using a teacher model to create the thought streams. Finally we use SFT to train the model. Further we use DPO to better align the model to periodically ask the member for permission to continue asking them questions (bedside manners?).

---

## Project Overview

AIDiT is a toolkit for preparing, training, and evaluating large language models (LLMs) for clinical diagnosis tasks. It provides:

- **Data preparation**: Scripts and notebooks to process raw doctor-patient conversations and generate datasets for supervised fine-tuning (SFT).
- **Model fine-tuning**: Notebooks and scripts for SFT, reward modeling, and DPO (Direct Preference Optimization) on clinical dialogue data. Our code is adapted (copied) from the Unsloth training code. We have currently fine tuned and DPOed the Qwen 3 model.
- **Evaluation**: Simulation and evaluation tools to benchmark model performance in doctor-patient scenarios.

---

## Directory Structure

```
aidit/
│
├── data/
│   ├── raw/         # Raw doctor-patient conversation transcripts (.txt)
│   ├── processed/   # Processed CSVs (e.g., doctor_patient_conversations.csv)
│   └── training/    # HuggingFace datasets for SFT and filtered variants
│
├── src/
│   ├── data_prep/   # Data preparation scripts and notebooks
│   ├── evals/       # Evaluation and simulation scripts
│   └── fine_tuning/ # Fine-tuning and reward modeling notebooks/scripts
│
└── README.md
```

---

## Data Preparation

- **Raw Data**: Place your raw conversation files in `data/raw/`.
- **Processing**: Use the notebooks in `src/data_prep/` (e.g., `Create SFT Data.ipynb`, `Create Training Data.ipynb`) to process raw data and generate training-ready datasets.
- **SFT Dataset Generation**: `sft_dataset_generator.py` automates the creation of SFT datasets from processed CSVs, with logic for conversation formatting and LLM prompt construction.

---

## Model Fine-Tuning

- **Notebooks**: The `src/fine_tuning/` directory contains Jupyter notebooks for:
  - SFT (Supervised Fine-Tuning) on clinical dialogue data
  - DPO (Direct Preference Optimization)
  - Reward modeling (`reward_functions.py`)
- **Datasets**: Use the datasets in `data/training/` for training. These are in HuggingFace format.

---

## Evaluation

- **Simulation**: `src/evals/driver.py` runs doctor-patient simulations, orchestrating interactions between model-based "Doctor", "Patient", and "Judge" agents.
- **Evaluation Scripts**: Other scripts in `src/evals/` (e.g., `doctor.py`, `patient.py`, `judge.py`) define the behavior of each agent.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
*(You may need to create this file based on your environment; typical dependencies include `openai`, `datasets`, `tqdm`, `pandas`, `numpy`, etc.)*

### 2. Prepare Data

- Place raw `.txt` files in `data/raw/`.
- Run the data preparation notebooks/scripts in `src/data_prep/` to generate processed CSVs and HuggingFace datasets.

### 3. Fine-Tune a Model

- Use the notebooks in `src/fine_tuning/` to fine-tune your chosen LLM on the prepared datasets.

### 4. Evaluate

- Run `src/evals/driver.py` to simulate doctor-patient dialogues and evaluate model performance.

---

## Customization

- **Prompts**: Modify prompt templates in `src/data_prep/prompts/` to experiment with different LLM prompting strategies.
- **Reward Functions**: Adjust reward logic in `src/fine_tuning/reward_functions.py` for custom RLHF or DPO training.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Inspired by open-source clinical NLP and LLM research.
- Uses [HuggingFace Datasets](https://huggingface.co/docs/datasets/) and [OpenAI API](https://platform.openai.com/docs/api-reference).
