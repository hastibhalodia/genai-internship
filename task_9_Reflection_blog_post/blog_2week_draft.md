# 2 Week Learnings — From Bigram to GPT-2

This week was all about language models — how they work and how they’ve evolved.

I started by building a Bigram Language Model using PyTorch. It’s a very basic model that predicts the next character based only on the previous one. Even though it’s simple, it helped me understand important concepts like tokenization, embeddings, and how models are trained using loss functions. I also learned how to get training and validation loss, and how to make the model generate sample text.

The next step was learning how to measure model performance. I implemented a perplexity calculator, which tells how well a model predicts text. I used both my custom function and the built-in `torchmetrics.Perplexity` to compare results. This gave me more confidence in evaluating models.

While I didn’t directly work with GPT-2, I read about it to understand the differences. Unlike the Bigram model, GPT-2 is based on transformers. It looks at longer parts of the text (not just the last character), which helps it make smarter predictions. It also has many layers and uses self-attention to understand context better. Learning about GPT-2 showed me how modern language models are built for tasks like translation, summarization, and chat.

### Key Takeaways
- Bigram models are simple but great for learning the basics.
- Perplexity is a useful metric to evaluate how good a model is.
- GPT-2 shows how powerful modern LMs can be by using attention and deeper networks.

Overall, Week 2 helped me build a strong foundation. I’m excited to move ahead and learn more about transformer-based models in the coming weeks.
