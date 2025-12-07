import argparse

from leanllm import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model ID from HuggingFace")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    args = parser.parse_args()

    llm = LLM(args.model, device=args.device)
    output = llm.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(output)


if __name__ == "__main__":
    main()
