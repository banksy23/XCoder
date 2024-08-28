## How Do Your Code LLMs Perform? Empowering Code Instruction Tuning with Really Good Data</h2>


<p>
📃 <a href="">ArXiv Paper</a>
  •
📚 <a href="https://huggingface.co/datasets/banksy235/XCoder-80K">Dataset</a>
    •
🕊 <a href="https://huggingface.co/banksy235/Xcoder-8B">XCoder-8B</a>
      •
🕊 <a href="https://huggingface.co/banksy235/Xcoder-70B">XCoder-70B</a>
      •
🐬 <a href="">Complexity Scorer</a>
        •
🐋 <a href="">Unit Test Model</a>
</p>

### ⭐ Introduction
There has been a growing interest in studying how to construct better code instruction tuning data. However, we find that many datasets, such as Magicoder-Evol-Instruct, suffer from severe data leakage. After cleaning up most of the leaked data, some well-known high-quality datasets perform poorly. This discovery reveals a new challenge: identifying which dataset genuinely qualify as high-quality code instruction data. We construct a data pool that includes almost all open-source code instruction fine-tuning datasets and proposed an efficient code instruction data selection strategy. We select good samples to train XCoder, which achieves new SOTA performance using fewer training data. Moreover, we perform a comprehensive analysis on the data composition and find existing code datasets have different characteristics according to their construction methods, which provide new insights for future code LLMs.

<details>
  <summary>Case Study on Data Leakage</summary>
<img width="591" alt="image" src="https://github.com/user-attachments/assets/25fdaf04-c9ca-4cf5-84d3-0fc640a93a56">

</details>

### Data Pool

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

### Data Selection For XCoder
<img width="585" alt="image" src="https://github.com/user-attachments/assets/e7c526a2-5488-45fe-9502-93c81b9e6756">

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

### New Insights For Code Instruction Data Synthesis
we analyze the data composition of XCoder, reassess the strengths and weaknesses of different data sources, and develop new insights into different data synthesis methods. Our conclusion can be summarized as follows:

- **instruction complexity**: data with more rounds has longer context and higher complexity. Additionally, Evol-Instruct is an effective method for improving instruction complexity.
- **response quality**: Data with added test case feedback verification during data synthesis tends to have higher quality. Furthermore, using a stronger model to synthesize data is a simpler, more direct, but effective approach.
- **instruction diversity**: We find that directly sampling from the real world and transforming it results in instructions with better diversity compared to other methods that only expand instructions using fixed seeds.
  
<img width="644" alt="image" src="https://github.com/user-attachments/assets/a0ae7eb3-7d73-407b-bb92-e1b576738d35">


### Citation
Please kindly cite our paper if it helps your research:
```bibtex
```
