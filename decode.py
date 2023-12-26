import torch
import torch.nn.functional as F

def greedy_search(model, prompt, tokenizer, vocab, max_len=50):
    """
    Generates a sequence of words using greedy decoding.

    Parameters
    ----------
    @param model: nn.Module
        The trained model
    @param prompt: str
        The prompt sequence. The sequence should be a string of space-separated tokens like "this is a prompt"
    @param vocab: torchtext.vocab.Vocab
        The vocabulary
    @param max_len: int
        The maximum length of the generated sequence
    """
    model.eval()

    # Convert the prompt sequence to a list of tokens
    prompt = vocab(tokenizer(prompt))

    with torch.no_grad():
        seq_len = len(prompt)
        print(seq_len)
        print("prompt sentence: ")
        print(" ".join(vocab.lookup_tokens(prompt)))
    
        # Initialize the output sequence with the actual target sequence
        output_sequence = prompt

        # Generate additional tokens using greedy decoding
        for _ in range(seq_len, max_len):
            input_data = torch.tensor(output_sequence[-seq_len:], dtype=torch.long).unsqueeze(0).T 
            # input_data.shape  seq_len, 1
            output_step = model(input_data)  
            # output_step.shape  seq_len, 1, vocab_size
            output_step = output_step.squeeze(1) # seq_len, vocab_size
            output_step_id = torch.argmax(output_step, dim=-1)[-1].item()
            output_sequence.append(output_step_id)

            # Break if the generated token is an end-of-sequence token
            if output_step_id == vocab["<eos>"]:
                break
        
        
        print("output sentence: ")
        print(" ".join(vocab.lookup_tokens(output_sequence)))

def beam_search(model, prompt, tokenizer, vocab, max_len=50, beam_width=5):
    """
    Generates a sequence of words using beam search decoding.

    Parameters
    ----------
    @param model: nn.Module
        The trained model
    @param prompt: str
        The prompt sequence. The sequence should be a string of space-separated tokens like "this is a prompt"
    @param tokenizer: Callable
        Tokenizer function
    @param vocab: torchtext.vocab.Vocab
        The vocabulary
    @param max_len: int
        The maximum length of the generated sequence
    @param beam_width: int
        Width of the beam for beam search
    """
    model.eval()

    # Convert the prompt sequence to a list of tokens
    prompt = vocab(tokenizer(prompt))

    with torch.no_grad():
        seq_len = len(prompt)

        # Initialize the beam with the actual target sequence
        beams = [{"tokens": prompt, "score": 0.0}]

        for _ in range(seq_len, max_len):
            new_beams = []

            for beam in beams:
                input_data = torch.tensor(beam["tokens"][-seq_len:], dtype=torch.long).unsqueeze(0).T
                output_step = model(input_data)
                output_step = output_step.squeeze(1)

                # Apply log softmax to the output probabilities
                log_probs = F.log_softmax(output_step, dim=-1)

                # Get the top-k candidates for the next token
                topk_values, topk_indices = torch.topk(log_probs[-1], beam_width)

                for value, index in zip(topk_values, topk_indices):
                    new_score = beam["score"] + value.item()
                    new_tokens = beam["tokens"] + [index.item()]

                    new_beams.append({"tokens": new_tokens, "score": new_score})

            # Sort the beams based on their scores
            new_beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)

            # Select the top-k beams to continue the search
            beams = new_beams[:beam_width]

            # Break if all beams have generated the end-of-sequence token
            if all(beam["tokens"][-1] == vocab["<eos>"] for beam in beams):
                break

        # Select the beam with the highest score
        best_beam = max(beams, key=lambda x: x["score"])

        print("input sentence: ")
        print(" ".join(vocab.lookup_tokens(prompt)))

        print("output sentence: ")
        print(" ".join(vocab.lookup_tokens(best_beam["tokens"])))
