## How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with Really Good Data</h2>


### 🗂️🗂️ Resources
We will open source all datasets, model checkpoints and codes, welcome everyone to follow us:

📃 Read our <a href="https://arxiv.org/pdf/2409.03810">Paper</a> on arxiv .

📚 Get our <a href="https://huggingface.co/datasets/banksy235/XCoder-80K">Dataset</a> on huggingface.

🕊 Try our Coder:  Get **XCoder-8B** from **huggingface** or <a href="https://modelscope.cn/models/banksy235/XCoder-8B">modelscope</a>.

🕊 Try our Coder: Get **XCoder-70B** form **huggingface** or <a href="https://modelscope.cn/models/banksy235/XCoder-70B">modelscope</a>.

🐬 We train a model to score the complexity of each instruction: Get **Complexity Scorer** from **huggingface** or <a href="https://modelscope.cn/models/banksy235/XCoder-Complexity-Scorer">modelscope</a>.

🐋 We trained a model to generate unit test programs for each candidate solution: Get **Unit Test Model** from **huggingface** or <a href="https://modelscope.cn/models/banksy235/Unit_Test_Model">modelscope</a>.

(The model weight uploading to Hugging Face is still in progress, and it will be open-sourced soon.)

---

### 😃 Motivations & Key Findings.
The performance of large language models on programming tasks is impressive, but many datasets suffer from data leakage, especially on benchmarks like HumanEval and MBPP. To address this, we introduce the **Test Leakage Indicator (TLI)**, which identifies high-leakage data, and clean it. We also evaluate on cleaner benchmarks, LiveCodeBench and BigCodeBench, using filtered data on LLaMA3.

Our findings reveal that some widely used datasets, like Magicoder-Evol-Instruct, are less reliable than previously thought. Inspired by alignment and mathematical data selection works, we select training data based on instruction complexity, code pass rate, and diversity. With just 40K examples, our model XCoder matches top performance and surpasses prior results at 80K.

Beyond cleaner data, we aim to redefine what makes a good Code Instruction Tuning dataset, analyzing previous works through XCoder's three key dimensions: [🎉🎉 New Insights For Code Instruction Data Synthesis](#-new-insights-for-code-instruction-data-synthesis).

<details>
  <summary>Click here, if you are curious about some leaked cases.</summary>
<img style="width: 100%;" alt="image" src="https://github.com/user-attachments/assets/25fdaf04-c9ca-4cf5-84d3-0fc640a93a56">

</details>

---
### Use TLI to detect the extent of data leakage in your training set.
```
python3 compute_TLI.py \
  --train_data_path {train_dataset} \
  --test_data_path {test_dataset} \
  --key_train {key name of the instruction in the training data JSON} \
  --key_test prompt {key name of the instruction in the test data JSON} \
  --only_analysis true
```
---
### 🌠 What open-source data do we collect?

We construct a data pool that includes many open-source code instruction fine-tuning datasets. The specific datasets are listed in the table below:
| Dataset                          | Data Size | Instruction Source          | Response Source      |
|----------------------------------|-----------|-----------------------------|----------------------|
| Code-290k-ShareGPT-Vicuna        | 289k      | -                         | -                  |
| CodeExercise-Python-27k          | 27k       | GPT                         | GPT                  |
| CodeUp                           | 19k      | GPT(Self-Instruct)                         | GPT                  |
| Glaive-code-assistant-v3         | 950k      |  Glaive              | Glaive               |
| oa_leet_10k                 | 23k       | -                         | -                  |
| Code-Alpaca                   | 20k       | GPT(Self-Instruct)          | GPT                  |
| Codefuse-Evol-Instruct         | 66k       | GPT(Evol-Instruct)                         | GPT             |
| DolphCoder    | 79k       | GPT(Evol-Instruct)               | GPT                  |
| Magiccoder-Evol-Instruct | 110k      | GPT(Evol-Instruct)         | GPT                  |
| MagicCoder-OSS-Instruct | 75k       | GPT(OSS-Instruct)                        | GPT                  |
| CommitPackFT                     | 702k      | GitHub       | GitHub               |
| StarCoder-Self-Align | 50k       | StarCoder2(OSS-Instruct)                        | StarCoder2               |
| Leet10k_alpaca                   | 10k       | -    | -             |

### 🔑 Data Selection Method For XCoder
<img src="https://github.com/user-attachments/assets/e7c526a2-5488-45fe-9502-93c81b9e6756" alt="Illustration of our data selection approach." style="width: 100%;">

XCoder selects good samples based on three dimensions: instruction complexity, response quality, and instruction diversity.

- **Instruction complexity**: People always hope that Code LLM can write more complex programs.Thus, we train a <a href="">Complexity Scorer</a> to measure the complexity of each sample.
- **Response quality**: We use the number of passed test cases as a measure of code coverage quality. We train a <a href="">Unit Test Model</a> to generate a unit test program for each sample. Compared to using language models directly to judge code correctness, executing test cases can obtain real-world feedback and have better judgment performance.
- **Instruction diversity**: As a general principle, an advanced LLM should be able to handle various requests from humans. We use Diversity-based Sampling method to ensure the diversity of the selected data.

---

### 🎖 Performance


| Dataset                  | Size   | LiveCodeBench Pass@1 | LiveCodeBench Easy-Pass@1            | BigCodeBench Pass@1 | HumanEval Base-Pass@1   |  HumanEval Plus-Pass@1         |     
|--------------------------|--------|---------------|-------------|--------------|--------------|-----------|
| Code-Alpaca              | 20k    | 0.0           | 0.0         |    11.9      | 30.5         | 25.6      |         
| StarCoder2-Self-Align    | 50k    | 9.5           | 24.7        |    14.5      | 37.8         | 34.8      |         
| Codefuse-Evol-Instruct*  | 66k    | 12.3          | 33.1        |    25.4      | 59.1         | 53.7      |          
| Magicoder-OSS-Instruct   | 75k    | 12.8          | 33.8        |    22.0      | 54.3         | 50.0      |     
| Magicoder-Evol-Instruct* | 100k   | 13.0          | 34.5        |    21.8      | 65.9         | 59.8      |    
| Code-Feedback            | 64k    | 14.8          | 38.0        |    27.0      | 56.7         | 51.8      |       
| **XCoder**               | 40k    | 16.5          | 43.7        |    27.4      | 54.9         | 50.6      |      
| **XCoder**               | 80k    | 16.8          | 43.7        |    29.6      | 57.3         | 53.0      |   

* \* means that the original dataset may have data leakage, and we perform a n-gram decontamination.

---

### 🎉🎉 New Insights For Code Instruction Data Synthesis
We analyze XCoder's data composition, reassess various data sources, and gain new insights into data synthesis. Our key findings:

- **Complexity**: Training models to assess instruction complexity outperforms heuristic methods. Evol-Instruct is effective for enhancing complexity, especially with longer, multi-round contexts.
- **Quality**: Test case execution provides better feedback for judging code correctness than model-based heuristics. Stronger models also yield higher-quality synthesized data.
- **Diversity**: Diverse instruction tuning is crucial. Real-world data sampling leads to better diversity than expanding instructions from fixed seeds.
<details>
  <summary>Click here, if you are curious about the data composition of XCoder</summary>
<img style="width: 100%;" alt="image" src="https://github.com/user-attachments/assets/a0ae7eb3-7d73-407b-bb92-e1b576738d35">
</details>

---

### Citation
Please kindly cite our paper if it helps your research:
```bibtex
@article{wang2024your,
  title={How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data},
  author={Wang, Yejie and He, Keqing and Fu, Dayuan and Gongque, Zhuoma and Xu, Heyang and Chen, Yanxu and Wang, Zhexu and Fu, Yujia and Dong, Guanting and Diao, Muxi and others},
  journal={arXiv preprint arXiv:2409.03810},
  year={2024}
}
```
