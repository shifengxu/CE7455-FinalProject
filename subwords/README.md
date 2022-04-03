# CE7455-FinalProject Subword Module.

## References

We utilize HuggingFace tokenizer package (more to come!):
  - [HuggingFace](https://github.com/huggingface/tokenizers) (Original implementation) <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="100"/>

## How to install HuggingFace tokenizer package

```bash
pip install tokenizers
```
![](./images/install-tokenizers.png)

## How to run how_to_run.py

In the "subwords/" folder:
```python
python -u how_to_run.py
```
![](./images/how_to_run.png)

## Sample input and output

Given the following input:
```python
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
```
The output will be:
```
['He', 'll', 'o', ',', 'y', "'", 'all', '!', 'How', 'are', 'you', '?']
```
