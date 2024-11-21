# OCR LLM - Summer 2024 OCR Correction Project

## Original Image Files
- **Path:**      `/data/lhyman6/OCR/data/images/*.jpg`

## Pytesseract Image Files
- **Path:**      `/data/lhyman6/OCR/data/images/*.txt.pyte`

## Chat-GPT Vision Files
- **Path:**      `/data/lhyman6/OCR/data/images/*.txt`


---
##BORR

### Rolls
Roll images are in /data/lhyman6/OCR/data/borr/BORR

ChatGPT reference: https://chatgpt.com/c/7d3e61b4-1ac4-4ac6-ab31-25039e73640c

To download use:


To expand use:
/data/lhyman6/OCR/scripts/borr/expand_rolls_pdf.py
---

## Experiment Design
The project aims to compare the training of LLMs for OCR with different levels of gold and silver quality data.

### Models and Data Sizes
- **BART**        (100, 1000, 10000)
- **LLAVA**       (untuned, 100, 1000, 10000)

### Nomenclature
- Example: `Bart, 100, Gold` = `bart_100_gold`

---

## Batch Operations
Using `ocrenv`:

- **Batch Upload:**     Send all the images to Chat 4o for processing.
- **Batch Download:**   Retrieve all the data.
- **Write CSV:**        Read the OCR text into a file called `silver_ocr_data.csv`.

---

## Bart Training
Train BART to correct OCR with both gold and silver data in increments of 100, 1000, and 10000.
- **Gold Data:**    `/data/lhyman6/OCR/data/gompers_corrections/bythepeople1.csv`
- **Silver Data:**  `/data/lhyman6/OCR/scripts/ocr_llm/silver_ocr_data.csv`

---

## BART Training Workflow

### Download Images
- **Script:**      `download_images.py`                (this needs to be modified for the second CSV file)

### OCR Images with Slurm
- **Scripts:**     `ocr_pyte.py` and `ocr_slurm.sh`    (processes images into Pytesseract)

### Preprocess the OCR Text
- **Script:**      `writecsv.py`                       (extracts data, cleans, reformats for LLM training)
- **Script:**      `read_data.py`                      (combines OCR and original data, cleans up)

### Tokenize the Text
- **Script:**      `tokenize_data.py`                  (labels tokens for training)

### Training
- **Script:**      `train_bart.py`                     (basic training script, runs on two GPUs)
- **Template:**    `train_bart_template.sh`            (basic template for the Slurm script)
- **Script:**      `generate_training_scripts`         (creates training scripts for different models)
- **Script:**      `submit_bart_training`              (submits all models for training)

### Plot Training Results
- **Script:**      `plot_bart_training`                (plots training results) *#untested*

---

## Testing
- **Script:**      `bart_test.py`                      (generates results)
- **Script:**      `bart_test_validate.py`             (uses validation tools)

### Process The OCR
- **Script:**      `run_bart_slurm.sh`                 (modifiable for more GPUs, processes OCR text)
- **Script:**      `/data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_1.py`  (tests current model, checks model list)

---

## LLAMA 3.2 Vision Training 
"meta-llama/Llama-3.2-11B-Vision-Instruct"

source l40senv/bin/activate
export TRANSFORMERS_CACHE=/data/lhyman6/OCR/scripts_newvision/llama
export HF_HOME=/data/lhyman6/OCR/scripts_newvision/llama


### Preprocess the OCR Text
- **Script:**      `writecsv.py`                       (extracts data, cleans, reformats for LLM training)
- **Script:**      `read_data.py`                      (combines OCR and original data, cleans up)



llama32.slurm       the slurm file for running the model

###**complete_testing_csv.csv  'contains the image filename, human transcirption, pytesseract transcription, and chatgpt transcription"
id	transcription	pyte_ocr	chatgpt_ocr

create_annotation_llama32.py        creates the json files that link the prompt (user), the answer (assistant), and the image together. it creates a train.json and a test.json.

train_llama32.py and train_llama32.slurm    trains the model. Still needs debugging.

1. make dataset file (convert images and texts to embeddings)
2. load datasets
run trainer on those datasets

https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaForConditionalGeneration

https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/31


https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/finetune_vision_model.md

CURRENT PROBLEM: NEED TO SET HF TOKENS IN TMP DIRECTORIES








# HOW TO RUN NOVEMBER


The BART models are run through these programs
The files that run the BART models are:
/data/lhyman6/OCR/scripts/ocr_llm/run_bart_models_slurm.sh
/data/lhyman6/OCR/scripts/ocr_llm/run_bart_models.py


Running the LLAMA 3.2 BASE
run  /data/lhyman6/OCR/scripts_newvision/llama/llama_l40s_production.py

Cleaning data: /data/lhyman6/OCR/scripts_newvision/llama/clean_llama32_text.py


Comparing DAta: /data/lhyman6/OCR/scripts_newvision/llama/compare_csv.py



*****

experimenting with a ollama bash script. process_images.sh


an6@login03 lhyman6]$ sbatch train_llama32_a100.slurm
sbatch: error: Unable to open file train_llama32_a100.slurm
[lhyman6@login03 lhyman6]$ cd /data/lhyman6/OCR/scripts_newvision/llama
[lhyman6@login03 llama]$ sbatch train_llama32_a100.slurm 
Submitted batch job 17978748
[lhyman6@login03 llama]$ 

    teh variations in usage should only be in the slurm configuration


    checkpoiunts are in l40s directory even though i am using ica100s

---
## HOW TRAINING WORKS
You create pairs of images and text, where the text should be the model response.



---
## LLAVA Training 1.6  llava-hf/llava-v1.6-mistral-7b-hf

Using `llavaenv`, need to build flash-attn and everything while on cuda gpu.

The deepspeed config file `zero3.json`  

### Data Preparation
- **Script:**      `llava_data_read.py`                (links images and texts into JSON format for LLAVA training, uses `complete_bart_training_data.csv` from BART training)
- **Output Folder:** `/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16/`

### Model Training
- **Script:**      `train_mem_16.py`                      (from [LLaVA GitHub](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train_mem.py))

**Finetune Instructions:** [LLaVA Finetune Documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)

Make sure you have:
1. `cuda/12.1.0`
2. Paths to `deepspeed` and path to `cutlass`

`train.py` is backed up locally by `move_train.py` but the version you need to edit in order to make changes is actually `"/home/lhyman6/.local/lib/python3.8/site-packages/llava/train/train.py"`

Needed to alter '/home/lhyman6/.local/lib/python3.8/site-packages/deepspeed/constants.py'  to use the MASTER_PORT for the distributed port address instead of an arbitrary number

### HOWTO
- **Script:**      `generate_llava_scripts_16.py`          (creates the versions of that for each type of training)
- **Script:**      `submit_llava_METAL_NUMBER.sh`       (submit these with `sbatch` to run training)

Checkpoint directories are located at: /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16/gold_100/checkpoints/llava-hf/llava-v1.6-mistral-7b-hf-task

Current issues: the training doesn't involve splitting dataset into training, validation, and testing sets like with BART.

I am going to train everything without doing that and see what happens.


---



## ANALYSIS

Now that we have models, we need to test the results. We need to construct a CSV file that looks like this:

| id | transcription | pyte_ocr | chatgpt_ocr | BART_untuned | BART_gold_100 | BART_gold_1000 | BART_gold_10000 | BART_silver_100 | BART_silver_1000 | BART_silver_10000 | LLAVA_untuned | LLAVA_gold_100 | LLAVA_gold_1000 | LLAVA_gold_10000 | LLAVA_silver_100 | LLAVA_silver_1000 | LLAVA_silver_10000 |
|----|---------------|----------|-------------|--------------|---------------|----------------|-----------------|-----------------|------------------|-------------------|---------------|----------------|-----------------|------------------|------------------|-------------------|--------------------|
|    |               |          |             |              |               |                |                 |                 |                  |                   |               |                |                 |                  |                  |                   |                    |
|    |               |          |             |              |               |                |                 |                 |                  |                   |               |                |                 |                  |                  |                   |                    |

- **Script:**   `download_images.py`                (this needs to be modified for the second CSV file)
- **Scripts:**  `ocr_pyte.py` and `ocr_slurm.sh`    (processes images into Pytesseract)

- **Scripts:**  `generate_complete_testing.py`     this script reads in the LOC CSV and then reads in additional data (pyte OCR and openai OCR) from the data directory
creates complete_testing_csv.csv

- **Scripts:**  'run_bart_models.py"    runs the trained models over the data to generate the text.
creates processed_testing_csv.csv


*

bart models completed.


LLAVA
/data/lhyman6/OCR/scripts/ocr_llm/run_llava_16_models.py   Runs the LLava 1.6 models to update the basic csv

