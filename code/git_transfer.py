import torch
import torch.cuda
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datasets

from textwrap import wrap
import matplotlib.pyplot as plt

from evaluate import load

from transformers import TrainingArguments, Trainer

from datetime import datetime

def get_torch_dataset(data_dir):
    ds = datasets.load_dataset("imagefolder", data_dir=data_dir, split="train")
    plot_images(ds['image'][:1], ds['text'][:1])
    ds = ds.train_test_split(test_size=0.1)
    train_ds = ds["train"]
    test_ds = ds["test"]
    return train_ds,test_ds

def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}



def train(device,processor,train_ds,test_ds):
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    model_name = 'git-base'
    model = model.to(device)
    def transforms(example_batch):
        images = [x for x in example_batch["image"]]
        captions = [x for x in example_batch["text"]]
        inputs = processor(images=images, text=captions, padding="max_length")
        inputs.update({"labels": inputs["input_ids"]})
        return inputs
    train_ds.set_transform(transforms)
    test_ds.set_transform(transforms)

    training_args = TrainingArguments(
                                    output_dir=f"{model_name}-sysml-{datetime.now().strftime("%d%H%M")}",
                                    learning_rate=5e-4,
                                    num_train_epochs=60,
                                    fp16=True,
                                    per_device_train_batch_size=10,
                                    per_device_eval_batch_size=10,
                                    gradient_accumulation_steps=2,
                                    save_total_limit=3,
                                    eval_strategy="steps",
                                    eval_steps=60,
                                    save_strategy="steps",
                                    save_steps=60,
                                    logging_steps=20,
                                    remove_unused_columns=False,
                                    label_names=["labels"],
                                    load_best_model_at_end=True,
                                    )
    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=test_ds,
                    compute_metrics=compute_metrics,
                    )
    return trainer

def test(device,processor,model,image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)


def main():
    data_dir = "data//im"
    checkpoint = "microsoft/git-base"
    train_ds,test_ds = get_torch_dataset(data_dir)
    wer = load("wer")
    processor = AutoProcessor.from_pretrained(checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = train(device,processor,train_ds,test_ds)
    trainer.train()

if __name__ == "__main__":
    main()


    