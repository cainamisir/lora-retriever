from datasets import load_dataset
ds = load_dataset("WaterWang-001/CodeFlowBench-2505", split="train")
ds.to_json("codeflow/data/codeflowbench_full.jsonl", lines=True, force_ascii=False)