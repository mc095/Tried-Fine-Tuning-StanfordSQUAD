# My Experience Fine-tuning FLAN-T5-small on SQuAD Dataset

This is my personal take on what happened when I evaluated my fine-tuned FLAN-T5-small model on a subset of the Stanford Question Answering Dataset (SQuAD). I used a custom Python script to fine-tune the model with Parameter-Efficient Fine-Tuning (PEFT) and LoRA, and the results gave me a lot to think about. Here’s a clear breakdown of my model’s performance, the gaps I noticed, and what I think went wrong.

## How I Set It Up

I fine-tuned the FLAN-T5-small model (~80M parameters) on a 1000-example chunk of SQuAD’s training data (`train[:1000]`). Here’s the setup I used:
- **Model**: `google/flan-t5-small`, loaded with my Hugging Face token.
- **Fine-tuning Approach**: PEFT with LoRA (`r=8`, `lora_alpha=32`, `lora_dropout=0.1`, targeting `q` and `v` modules).
- **Training Settings**:
  - Batch size: 8 (with 4 gradient accumulation steps).
  - Learning rate: 1e-4.
  - Epochs: 3.
  - No validation (`eval_strategy="no"`).
- **Evaluation**:
  - **Test Set**: 100 examples from the same 1000-example subset (`dataset.select(range(100))`).
  - **Original Test Examples**: 5 handpicked question-context pairs about Notre Dame’s architecture and Catholic history.
  - **Metrics**: Exact Match (EM), F1 score, and accuracy, calculated manually because I couldn’t get `evaluate.load("squad")` to work.

I stuck to just my original variables: `peft_model`, `tokenizer`, `dataset`, and `device`.

## My Results

### Test Set Results (100 Examples)
- **Exact Match (EM)**: 50.00%
  - Half of my model’s answers matched the reference answers exactly.
- **F1 Score**: 65.00%
  - This showed my model often got some words right, even if the full answer wasn’t perfect.
- **Accuracy**: 50.00%
  - Same as EM, since I defined accuracy as exact string matches (ignoring case and words like “the”).

### Original Test Examples (5 Questions)
- **Accuracy**: 20.00% (1/5 correct)
- **Breakdown**:
  1. **Question**: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes, France?
     - **Context**: Mentions the Grotto as a replica of Lourdes.
     - **Expected**: Saint Bernadette Soubirous
     - **Model**: Mary
     - **What Happened**: Total miss. My model gave a vague answer instead of the specific name.
  2. **Question**: What is in front of the Notre Dame Main Building?
     - **Context**: Talks about the Main Building’s gold dome.
     - **Expected**: a copper statue of Christ
     - **Model**: the main building
     - **What Happened**: Wrong and confusing. My model thought the building was in front of itself.
  3. **Question**: The Basilica of the Sacred Heart at Notre Dame is beside to which structure?
     - **Context**: Says the Basilica is beside the Main Building.
     - **Expected**: the Main Building
     - **Model**: Main Building
     - **What Happened**: Got this one right after I fixed a bug that missed matches due to “the”.
  4. **Question**: What is the Grotto at Notre Dame?
     - **Context**: Describes the Grotto as a Marian place of prayer and reflection.
     - **Expected**: a Marian place of prayer and reflection
     - **Model**: a replica of the grotto
     - **What Happened**: Off the mark. My model latched onto the Grotto’s origin, not its purpose.
  5. **Question**: What sits on top of the Main Building at Notre Dame?
     - **Context**: Mentions a golden statue of the Virgin Mary on the gold dome.
     - **Expected**: a golden statue of the Virgin Mary
     - **Model**: gold dome
     - **What Happened**: Mixed up the dome with the statue on it.

## What I Learned from the Results

### Test Set Performance (50.00% EM, 65.00% F1)
I was kinda pleased with the 50% EM—it showed my model could nail some answers, probably the simpler factual ones. The 65% F1 score told me it often got close, picking up key words (like “gold dome” instead of “golden statue of the Virgin Mary”), but it tripped on exact wording. Since my test set came from the same 1000 examples I trained on, I suspect the model got too cozy with those, making the results look better than they really are. The first 100 examples might’ve been easier or less varied too. The small size of FLAN-T5-small helped me train on my setup, but I think its limited capacity held it back from handling tougher questions.

### Original Test Examples Performance (20.00% Accuracy)
The 20% accuracy here was a gut punch—only 1 out of 5 questions right. My model really struggled with these specific questions I picked. Here’s what I noticed:
- **Historical Names**: Spitting out “Mary” instead of “Saint Bernadette Soubirous” was a big miss. I think my 1000-example dataset didn’t have enough specific names like that.
- **Spatial Stuff**: Getting “the main building” for what’s in front of it or “gold dome” for what’s on top showed my model didn’t get spatial relationships at all.
- **Context Mix-ups**: For the Grotto question, it focused on “replica of the grotto” instead of its role as a prayer spot, so it wasn’t fully processing the context.
- **Bug Fix**: I had a dumb bug where “Main Building” wasn’t counted as matching “the Main Building”. Fixing that by ignoring “the” bumped my accuracy from 0% to 20%, which was a relief.

### Why the Test Set and Original Examples Differed
The 50% EM on the test set versus 20% on my five questions was a head-scratcher. I think the test set did better because it was pulled from the training data, so the questions were familiar to the model. My five handpicked questions were tougher, hitting on specific details like historical figures and Notre Dame’s layout, which my dataset probably didn’t cover well. I might’ve overfit to the 1000 examples, making the model great on similar stuff but terrible on new, curated questions. Those five questions also tested different skills (names, spatial, functional), which exposed my model’s gaps.

## What I Did Wrong

Looking back, I made some mistakes that tanked my model’s performance, especially on those five test questions:

1. **Tiny Training Dataset**:
   I only used 1000 examples from SQuAD’s training set, which is way smaller than the full ~87,000. That meant my model missed out on a ton of variety in questions and contexts, like specific names (e.g., Saint Bernadette) or spatial details. My laptop’s limits forced me to keep it small, but it hurt the results.

2. **No Validation**:
   I set `eval_strategy="no"`, so I didn’t check how the model was doing during training. That was a bad call—it probably overfit to my 1000 examples without me noticing. I also didn’t tweak things like learning rate or epochs because I had no validation feedback.

3. **Small Model**:
   FLAN-T5-small was great for my setup (16GB RAM), but its ~80M parameters couldn’t handle SQuAD’s complexity well. Using LoRA helped, but the low rank (`r=8`) might’ve limited how much the model could adapt.

4. **Bad Test Set Choice**:
   I pulled my 100 test examples from the training data, which was a mistake. It made my model look better (50% EM) than it actually was. My five curated questions were more like a real test, and they showed how weak the model really was.

5. **Metric Problems**:
   I couldn’t get `evaluate.load("squad")` to work because of some Hugging Face Hub issue, so I had to hack together my own EM and F1 calculations. My F1 score is a bit rough since it’s just token overlap, not the official SQuAD metric. Plus, I had that string comparison bug that missed “Main Building” matches at first.

6. **Generation Settings**:
   I set `max_new_tokens=50` for generating answers, which might’ve cut off longer answers (like “golden statue of the Virgin Mary” turning into “gold dome”). My model also leaned toward short, vague answers (e.g., “Mary”), probably because it wasn’t trained enough or couldn’t process contexts fully.

## How I Can Fix It

I want to stick to my variables (`peft_model`, `tokenizer`, `dataset`, `device`), so here’s what I’m planning to do:

1. **Use More Data**:
   If my laptop can handle it, I’ll try a bigger SQuAD chunk, like `train[:5000]`:
   ```python
   dataset = load_dataset("squad", split="train[:5000]", token=token)
   ```
   That should give my model more variety to learn from.

2. **Split My Dataset**:
   I’ll divide my 1000 examples into 900 for training and 100 for testing:
   ```python
   train_dataset = dataset.select(range(900))
   test_dataset = dataset.select(range(900, 1000))
   tokenized_train = train_dataset.map(preprocess_function, batched=True)
   tokenized_test = test_dataset.map(preprocess_function, batched=True)
   ```
   This way, I’m not testing on stuff I trained on.

3. **Train Longer**:
   I’ll bump up to 5 epochs:
   ```python
   training_args = TrainingArguments(
       ...,
       num_train_epochs=5,
       ...
   )
   ```
   Hopefully, that’ll help my model learn better.

4. **Tweak Answer Generation**:
   I’ll let the model generate longer answers:
   ```python
   outputs = peft_model.generate(
       input_ids=input_ids,
       max_new_tokens=100,
       do_sample=False
   )
   ```

5. **Check Context Handling**:
   My model’s messing up on context (e.g., “gold dome” instead of the statue). I’ll test with longer contexts or rephrase questions to make key details stand out.

6. **Add Some Validation**:
   If I can spare the resources, I’ll use 100 examples for validation:
   ```python
   train_dataset = dataset.select(range(900))
   val_dataset = dataset.select(range(900, 1000))
   trainer = Trainer(
       ...,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       ...
   )
   training_args = TrainingArguments(
       ...,
       eval_strategy="epoch",
       ...
   )
   ```

## Wrapping Up

My FLAN-T5-small model did okay on a 100-example test set (50% EM, 65% F1) but bombed on my five curated questions (20% accuracy). The big problems were my tiny 1000-example dataset, skipping validation, using a small model, and testing on training data. My five questions showed my model struggles with historical names, spatial details, and getting the context right. I’m planning to use more data, split my dataset, train longer, and tweak how answers are generated—all while sticking to my variables. If I had more computing power, I’d try a bigger model or the full SQuAD dataset, but for now, this project taught me a lot about the limits of fine-tuning a small model with limited data.
