## How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with Really Good Data</h2>


### 🗂️🗂️ Resources
📃 Read our <a href="https://arxiv.org/pdf/2409.03810">Paper</a>.

📚 Get our <a href="https://huggingface.co/datasets/banksy235/XCoder-80K">Dataset</a> on huggingface.

🕊 Try our Coder:  Get <a href="https://huggingface.co/banksy235/Xcoder-8B">XCoder-8B</a> on huggingface.

🕊 Try our Coder: Get <a href="https://huggingface.co/banksy235/Xcoder-70B">XCoder-70B</a> on huggingface.

🐬 We train a model to score the complexity of each instruction: Get <a href="">:Complexity Scorer</a> on huggingface.

🐋 We trained a model to generate unit test programs for each candidate solution: Get <a href="">Unit Test Model</a> on huggingface.



### 😃😃 Quickly understand our motivations and conclusions.
The performance of large language models on programming tasks is surprising. More and more work is being done to study how to obtain a good Code Instruction Tuning training set, and their effectiveness is evaluated on HumanEval and MBPP. However, we found that many of these works suffer from serious data leakage issues on HumanEval and MBPP, indicating that the performance gains achieved by these methods are likely due to data leakage. Furthermore, a more serious problem is that we may need to redefine our understanding of what constitutes a good Code Instruction Tuning dataset.

To address the issue of data leakage, we propose the **T**est **L**eakage **I**ndicator (TLI), which measures the similarity between training and test datasets in terms of instructions. We used TLI to clean the training data. In addition, we introduce two new benchmarks, LiveCodeBench and BigCodeBench, which were proposed after the training data, reducing the possibility of contamination. We retrained and evaluated using the cleaned data on LLaMA3. We found that many training data are no longer as good as previously thought, such as Magicoder-Evol-Instruct, which was previously used as a foundation for many code and agent-related works due to its excellent performance on code generation tasks.

After data cleaning, we found that we lost the judgment of what constitutes a good Code Instruction Tuning dataset. Therefore, we decided to collect all available Code Instruction Tuning data and use it to filter out the best version of the training data. We referenced many alignment and mathematical works and decided to filter the data based on three aspects: instruction complexity, code pass rate, and instruction diversity. We trained our model, XCoder, using the filtered data. XCoder achieves a comparable level of performance as the best training data we collected with only 40K data, further surpassing all previous works at 80K.

In addition to obtaining a cleaner and more effective training data, we also hope to redefine our understanding of the construction methods for a good Code Instruction Tuning training set. To achieve this, we analyze the training data constructed by previous works from the three dimensions of XCoder. Our findings can be find at [🎉🎉 New Insights For Code Instruction Data Synthesis](#-new-insights-for-code-instruction-data-synthesis).

<details>
  <summary>Click here, if you are curious about some leaked cases.</summary>
<img style="width: 100%;" alt="image" src="https://github.com/user-attachments/assets/25fdaf04-c9ca-4cf5-84d3-0fc640a93a56">

</details>


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

- **instruction complexity**: People always hope that Code LLM can write more complex programs.Thus, we train a <a href="">Complexity Scorer</a> to measure the complexity of each sample.
- **response quality**: We use the number of passed test cases as a measure of code coverage quality. We train a <a href="">Unit Test Model</a> to generate a unit test program for each sample. Compared to using language models directly to judge code correctness, executing test cases can obtain real-world feedback and have better judgment performance.
- **instruction diversity**: As a general principle, an advanced LLM should be able to handle various requests from humans. We use Diversity-based Sampling method to ensure the diversity of the selected data.



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

### 🎉🎉 New Insights For Code Instruction Data Synthesis
we analyze the data composition of XCoder, reassess the strengths and weaknesses of different data sources, and develop new insights into different data synthesis methods. Our conclusion can be summarized as follows:

- **Complexity**: Training a language model to determine the complexity of each instruction is more accurate than heuristic rules such as length or perplexity. From a data synthesis perspective, data with more rounds has longer context and higher complexity. Additionally, Evol-Instruct is an effective method for improving instruction complexity.
- **Quality**: Code LLMs that deliver accurate responses. Compared to using language models directly to judge code correctness, executing test cases can obtain real-world feedback and have better judgment performance. Data with added test case feedback verification during data synthesis also tends to have higher quality. Furthermore, using a stronger model to synthesize data is a simpler, more direct, but effective approach.
- **Diversity**: To effectively cater to diverse human requests, an advanced language model must be versatile. Therefore, it is essential that the data used for its instruction tuning is as varied as possible. We have found that directly sampling from the real world (pre-training data) and transforming it results in instruction datasets with better diversity compared to other methods that only expand instructions using fixed seeds.
<details>
  <summary>Click here, if you are curious about the data composition of XCoder</summary>
<img style="width: 100%;" alt="image" src="https://github.com/user-attachments/assets/a0ae7eb3-7d73-407b-bb92-e1b576738d35">
</details>

### Citation
Please kindly cite our paper if it helps your research:
```bibtex
@misc{wang2024codellmsperformempowering,
      title={How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with High-Quality Data}, 
      author={Yejie Wang and Keqing He and Dayuan Fu and Zhuoma Gongque and Heyang Xu and Yanxu Chen and Zhexu Wang and Yujia Fu and Guanting Dong and Muxi Diao and Jingang Wang and Mengdi Zhang and Xunliang Cai and Weiran Xu},
      year={2024},
      eprint={2409.03810},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2409.03810}, 
}
```
