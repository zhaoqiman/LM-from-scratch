# LM-from-scratch
Midterm Assignment for AI-design course.

This project refers to [nanoGPT](https://github.com/karpathy/nanoGPT).
- Train a language model.
- Able to generate coherent and unambiguous sentences.


## Model Architecture
- **Embedded Layer**: The model begins with two embedding layers.
  - `wte`: for token embeddings   that convert input tokens into dense vectors.
  - `wpe`: for position embeddings that encode the position of each token in the sequence.

- **Transformer Blocks**: Core of the architecture consists of multiple Transformer blocks. Each block is composed of:
   - **Causal Self-Attention**: implements the attention mechanism that allows the model to focus on past tokens when predicting the next token in a sequence. It includes:
      -  Multi-head attention, where multiple attention mechanisms are run in parallel.
     -  A dropout layer for regularization, helping to prevent overfitting during training.
   - **Layer Normalization**: Applied before and after the attention and feedforward layers to stabilize the learning process.
   - **Feedforward Network (MLP)**: A fully connected feedforward network that processes the output of the attention layer, applying a GELU activation function and dropout for regularization.

- **Output Layer**: Finally, a linear layer `lm_head` maps the output embeddings back to the vocabulary size, allowing the model to predict the next token in the sequence.

## Dataset
tianji-etiquette-chinese-v0.1.json, from [tianji-chinese](https://huggingface.co/datasets/sanbu/tianji-chinese). A dataset on banquet etiquette in Chinese. Conversation history with Tianji-AI.
- **Preprocessing**: Conversation part of the raw json file was extracted for training. With 'prepare.py'.
    ``` sh
    python data/preprocess.py
    ```
- **Tokenization**: The dataset is tokenized using the GPT2 tokenizer, split into 90% training and 10% validation sets.
    ``` sh
    python data/prepare.py
    ```


## Training
**Parameter/Configs**: Defined in file 'training_jingjiu.py'.

```
python train.py config/train_jingjiu.py
```

Models are saved in the `out-jingjiu` directory. Total 5000 iters, checkpoint every 1000 iters.

## Generation
Generate some sentences with checkpoint 4000.
```
python sample.py --out_dir=out-jingjiu/4000
```

We set `start = "\n"` to start the generation with a newline character, `max_new_tokens = 500` to limit the length of the generated text to see generated text of ckpt-4000.

**Output**:
```

对于不太熟悉的领导，可以通过以下方式表现：1.如果对方讲话，可以适时地适当接赞美，比如「我现在您手下工作，咱们次有机会。」2.转移话题，将注意力引向其他人或其他方面支持，例如「我现在某某某某某方面，就是有什么心，非常感谢您的某某部门我们的非常敬意。」3.对于陌生人，可以表达对领导的感激和尊重，例如「很荣幸能和大家一起吃饭，今天有机会跟你一起吃饭，也有幸和大家一起共事这个项目的认识和支持。这样的话语不仅表达了对领导的感激，也显得自己的自然。
酒桌上被敬酒时，可以采
---------------

在酒局上敬酒时，首先要遵循‘先’的原则，即先上后下‘先上后下’的原则。其次，从右手拿起配，即先敬酒，先敬自己的距离。这样做既能展现你的礼貌，也能展示你的专业素养和对饭局的重视。
敬酒时，首先要注意以下几点：首先，通常先敬酒时机，不要随意掉规则；其次，不要直接说出不得不起的话，这样可以避免让对方感到尴尬或不适。在饭局上，要注意言辞的等程度，不要过于私语；最后，要注意邀请人同道菜肴的人，不要过于紧张，这些都可能会让客人感觉到你不重视这次聚会的压力。
要让自己添�
---------------

领导在饭局上让你提醒你可以采取以下步骤：首先，喝酒前先先表示歉意，比如提出替代方案。其次，可以准备一些语气和简短的话题，如“咱就深，您这杯酒我敬您，祝您工作顺利。”这样的话语不仅表达了对领导的尊重，也展示了你对领导的尊重和感激之情。
有助于提升自己的成就和关系的重要性。可以用轻松的语气氛，比如说：“领导，也这杯酒我敬您，祝您己工作顺利。”这样的话语既能展现你的聪明才智和对领导的尊重，也能表现出你的职场智慧和对未来的积极态度。
敬酒时应注意以下几点：�

```

