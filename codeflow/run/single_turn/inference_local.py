import os
import json
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.local import ChatModel
from src.utils import (
    get_filenames_without_extension,
    extract_code,
    get_input_single,
    ensure_python_code_block,
)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # Support both JSON array and JSONL inputs
    if args.input_file.endswith(".jsonl"):
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        data = json.load(open(args.input_file))

    chat_model = ChatModel(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_ray=args.use_ray,
    )
    filename_list = get_filenames_without_extension(args.output_dir)

    for problem in data:
        problem_description_now = problem["problem-description"]
        subproblems = problem["subproblems"]
        problemid = problem["problem-id"]
        overall_turns = problem["overall-turns"]

        # Skip already processed problems
        if problemid in filename_list:
            continue

        history = ""
        turn_number = 1
        for subproblem in subproblems:
            # Construct the prompt based on turn number and overall context
            user_input = get_input_single(subproblem, turn_number, overall_turns, problem_description_now, history)
            turn_number += 1

            # Accumulate context until final turn
            if turn_number != overall_turns + 1:
                history += "\n" + user_input
                continue

            # Final turn: complete generation
            input_all = [
                {"role": "system", "content": "You are a Programming Expert. You always provide correct and reliable code solutions."},
                {"role": "user", "content": user_input}
            ]
            generated = chat_model.generate(
                input_all,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output = generated[0].outputs[0].text

            # Ensure the output is wrapped in a code block
            output = ensure_python_code_block(output)
            subproblem.update({"original_output": output})

            # Extract code from generated output
            code_output = extract_code(output)
            subproblem.update({"prompt": user_input})
            subproblem.update({"generated": code_output})

        with open(f"{args.output_dir}/{problemid}.json", "w") as f:
            json.dump(problem, f, ensure_ascii=False, indent=4)

        print(f"Finished: saved to {args.output_dir}/{problemid}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-turn code generation inference")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=5120, help="Max model length for vLLM engine")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--dtype", type=str, default="auto", help="vLLM dtype (auto, float16, bfloat16, float32)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM GPU memory utilization fraction")
    parser.add_argument("--use_ray", action="store_true", help="Use Ray backend for vLLM (default: multiprocessing backend)")

    args = parser.parse_args()
    main(args)
