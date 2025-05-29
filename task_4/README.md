# Task 4: Hello LLM Quick-Start

## Overview

This task demonstrates how to use Hugging Face’s `transformers` pipeline to generate text with pretrained language models.

I've used the `"text2text-generation"` pipeline with the `google/flan-t5-base` model to generate a simple explanation.

## Code Summary

- Load the `"text2text-generation"` pipeline with the `flan-t5-base` model.
- Provide a prompt:  
  `"Explain generative AI to a 12-year-old in three sentences."`
- Generate and print the output text.

## Notes

- You can switch to other models like gpt2-medium by changing the pipeline and model.

- Adjust max_length and num_return_sequences to control output length and variety.

