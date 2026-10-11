---
license:
- mit
- other
license_link: https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md
library_name: transformers
language: en
tags:
- biology
- esm
- protein
- protein-language-model
- protein-embeddings
- masked-language-modeling
- transfer-learning
- variant-effect-prediction
- protein-engineering
- transformers
---

# ESMC

## Model Details

ESMC is a state-of-the-art protein language model that has learned the rules of protein biology from training on billions of protein sequences. ESMC provides representations of proteins enabling novel AI applications from therapeutic protein engineering to unlocking basic insights into protein biology across life.

The ESMC 6B model has 6 billion parameters, with 80 layers and 2.37e23 training flops. We additionally release overtrained 300M and 600M parameter variants of ESMC for local inference and finetuning.

The [ESMFold2](https://huggingface.co/biohub/ESMFold2) structure prediction models are trained on top of a frozen ESMC 6B language model. ESMFold2 is a state-of-the-art model for protein structure prediction and design that defines a new frontier for speed and accuracy.

The [ESMC sparse autoencoder](https://huggingface.co/biohub/ESMC-6B-sae-layer60-k64-codebook16384), `ESMC-6B-sae-layer60-k64-codebook16384`, is built on the ESMC 6B model and provides human-interpretable, agent-generated feature descriptions. See the [ESMC SAE overview card](https://huggingface.co/biohub/ESMC-SAE-Overview) for the full set of ESMC SAE variants.

To run this model with the Biohub Platform API, visit the [Biohub Platform](https://biohub.ai/).

Read more about ESMC in our paper [here](https://biohub.ai/papers/esm_protein.pdf).

### Example Usage

Install `esm` from GitHub (a PyPI release is coming soon):

```
pip install esm@git+https://github.com/Biohub/esm.git@main
```

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

# optionally use "biohub/ESMC-600M" or "biohub/ESMC-300M"
model = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-6B", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

inputs = tokenizer(GFP, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

print(f"logits shape: {tuple(output.logits.shape)}")
```

By default, the model returns only the final layer representations. To return hidden states from **all transformer layers**, set:

```py
output = model(**inputs, output_hidden_states=True)
```

For detailed usage, refer to the [Usage section below](#usage).

### Citation

```
@misc{candido2026language,
  title  = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan
            and Lin, Zeming and Verkuil, Robert and Wu, Bryan and Lee, Jin Sub
            and Bruguera, Elise S. and Keval, Jehan A. and Kopylov, Mykhailo
            and Pak, John E. and Wu, Wesley and Thomas, Neil and Mataraso, Samson
            and Hsu, Alvin and Trotman-Grant, Ashton C. and Fatras, Kilian
            and dos Santos Costa, Allan and Badkundri, Rohil and Ak{\i}n, Halil
            and Oktay, Deniz and Deaton, Jonathan and Montabana, Elizabeth
            and Sitwala, Hrishita and Yu, Yue and Wiggert, Marius
            and Carlin, Dylan Alexander and Goering, Anthony W. and Blazejewski, Tomasz
            and Sandora, McCullen and Hla, Michael and Jia, Tina Z.
            and Kloker, Leon H. and Sofroniew, Nicholas J. and Uehara, Masatoshi
            and Pannu, Jassi and Bachas, Sharrol and Liu, Daniel S.
            and Sercu, Tom and Rives, Alexander},
  year   = {2026},
  url    = {https://biohub.ai/papers/esm_protein.pdf},
  note   = {Preprint}
}
```

### Model Architecture

ESMC is based on the transformer architecture. It features Pre-LN, rotary embeddings, and SwiGLU activations. No biases are used in linear layers or layer norms.

### Parameters

ESMC was trained at multiple scales:

| Model | Parameters | Layers | Training FLOPs |
| :---- | ----: | ----: | ----: |
| **ESMC-300M** | 300M | 30 | 1.26e22 |
| **ESMC-600M** | 600M | 36 | 2.17e22 |
| **ESMC-6B** | 6B | 80 | 2.37e23 |

![][pal]

### Model Variants

| Model Variant | Description | URL |
| :---- | :---- | :---- |
| ESMC 300M | Smallest variant, publicly released. | [https://huggingface.co/biohub/ESMC-300M](https://huggingface.co/biohub/ESMC-300M) |
| ESMC 600M | Medium variant, publicly released. | [https://huggingface.co/biohub/ESMC-600M](https://huggingface.co/biohub/ESMC-600M) |
| ESMC 6B | Large variant, publicly released. | [https://huggingface.co/biohub/ESMC-6B](https://huggingface.co/biohub/ESMC-6B) |

### System Requirements

- Compute Requirements: GPU
- PyTorch environment with GPU support recommended.
- Recommended optional libraries: transformer\_engine, xformers

## Training Data

ESMC was trained on protein sequences from UniRef, MGnify, and the Joint Genome Institute (JGI). Sequence data was clustered at 70% sequence identity, resulting in 83M, 372M, and 2B clusters for UniRef, MGnify, and JGI, respectively.

### Training Procedure

Training was conducted in two stages:

- Stage 1: For the first 1 million steps, the model used a context length of 512, with metagenomic data constituting 64% of the training dataset.
- Stage 2: In the final 500,000 steps, the context length was increased to 2048, and the proportion of metagenomic data was reduced to 37.5%.

## Performance Metrics

Performance metrics are detailed in our [ESMC & ESMFold2 paper](https://biohub.ai/papers/esm_protein.pdf).

## Usage

### Flash Attention

Instead of scaled dot product attention (sdpa) you can use a flash attention backend. This requires running the model in bfloat16.

```py
model = (
    AutoModelForMaskedLM.from_pretrained(
        "biohub/ESMC-6B",
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    .to(torch.bfloat16)
    .eval()
)
```

### Sparse Autoencoder (SAE)

To get interpretable features from ESMC 6B hidden states and per-layer MLP outputs, you can choose from our pretrained SAEs. We provide the follow three:

* [ESMC SAEs for hidden states (all layers)](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers)
* [ESMC SAEs for one layer (different sparsity/codebook size)](https://huggingface.co/collections/biohub/esmc-saes-for-one-layer-different-sparsity-codebook-size)
* [ESMC SAEs for MLP outputs (all layers)](https://huggingface.co/collections/biohub/esmc-saes-for-mlp-outputs-all-layers)

```py
import torch
from transformers import AutoModel, AutoTokenizer

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

model = AutoModel.from_pretrained("biohub/ESMC-6B", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

sae_models = []
sae = AutoModel.from_pretrained(
    "biohub/ESMC-6B-sae-sweep-layer60-k64-codebook16384", device_map="auto"
)
sae_models.append(sae)

model.add_sae_models(sae_models)

inputs = tokenizer(GFP, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

print(f"num SAE outputs: {len(output.sae_outputs)}")
for i, sae_out in enumerate(output.sae_outputs):
    print(f"  SAE[{i}]: {type(sae_out).__name__}")
```

### Masked Language Modeling

ESMC can predict masked amino acids and compute the corresponding loss:

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
masked_GFP = "<mask>SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

model = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-6B", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

inputs = tokenizer(masked_GFP, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

labels = tokenizer(GFP, return_tensors="pt")["input_ids"].to(model.device)
# Only the masked positions contribute to the loss; everything else gets the
# ``-100`` ignore-index that ``CrossEntropyLoss`` skips.
labels = torch.where(inputs["input_ids"] == tokenizer.mask_token_id, labels, -100)

with torch.inference_mode():
    output = model(**inputs, labels=labels)

print(f"Loss: {output.loss.item():.6f}")
```

### Fine-tuning with peft

```py
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("biohub/ESMC-6B", device_map="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    target_modules=["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Attention Maps

To extract attention maps, pass `output_attentions=True`. Note: this is incompatible with `attn_implementation="flash_attention_2"`.

```py
output = model(**inputs, output_attentions=True)
# output.attentions: tuple of (batch, n_heads, seq_len, seq_len) tensors, one per layer
```

`output_attentions=True` triggers a manual, unoptimized attention path to extract the attention maps, which will reduce inference speed.

### Other Usage

You can access the base model without the pretrained LM head:

```py
import torch
from transformers import AutoModel, AutoTokenizer

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

model = AutoModel.from_pretrained("biohub/ESMC-6B", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

inputs = tokenizer(GFP, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

print(f"last_hidden_state shape: {tuple(output.last_hidden_state.shape)}")
```

Or use ESMC for Token Classification:

```py
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

model = AutoModelForTokenClassification.from_pretrained(
    "biohub/ESMC-6B", device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

inputs = tokenizer(GFP, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

predicted_token_class_ids = output.logits.argmax(-1)
predicted_tokens_classes = [
    model.config.id2label[t.item()] for t in predicted_token_class_ids[0]
]
print(f"logits shape: {tuple(output.logits.shape)}")
print(f"first 8 predicted classes: {predicted_tokens_classes[:8]}")
```

or Sequence Classification:

```py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

model = AutoModelForSequenceClassification.from_pretrained(
    "biohub/ESMC-6B", device_map="auto", num_labels=2
).eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-6B")

inputs = tokenizer(GFP, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

print(f"logits shape: {tuple(output.logits.shape)}")
```

For Token or Sequence Classification, the classifier head is not pretrained but instead meant to be fine-tuned for your downstream task.

## Frontier Safety

Biohub has established a safety team to assess the benefits and potential risks of our models and tools prior to release, and develop mitigations where necessary. Informed by our risk assessments, we are releasing the source code and model weights for ESMC 6B, ESMFold2, and ESMC SAEs. We are also releasing our ESM Atlas dataset and binder design system openly.

Prior to release, we conducted evaluations to inform our understanding of capability uplift for specific misuse-relevant functional tasks. The full details of these evaluations are available in our corresponding paper appendix.

[Biohub.ai](http://Biohub.ai) Platform: We implement guardrails that detect and restrict the use of keywords and sequences corresponding to controlled pathogens and toxins on our freely accessible platform. For further details regarding these guardrails, please refer to our Biohub platform Resources page.

## Biases and Limitations

### Potential Biases

- **Dataset bias:** Over- or under-representation of taxa, protein families, or ecological niches in public sequence and structure databases influences generalization and can bias outputs. This is partially mitigated by clustering-based, nonredundant sampling.

### Limitations

- **Context window:** ESMC has a context window limit of 2048 tokens.
- **Reliance on in-silico metrics:** Computational metrics do not replace wet-lab validation.

### Out-of-Scope or Unauthorized Use Cases

Do not use the model for the following purposes:

- Any use that is prohibited by the [Acceptable Use Policy](https://biohub.org/acceptable-use-policy/).

### Caveats and Recommendations

- Review and validate outputs generated by the model.
- We are committed to advancing the responsible development and use of artificial intelligence.
- Should you have any security or privacy issues or questions related to the services, please reach out to our team at [support@biohub.org](mailto:support@biohub.org).

[pal]: images/contact_pal.png
