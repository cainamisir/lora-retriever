import os
import json
import datasets
import ast
import sys
from tqdm import tqdm

# Add MemoryBench to path
sys.path.append(os.path.join(os.getcwd(), "MemoryBench"))
from src.evaluate import evaluate_and_summary
from src.utils import get_single_dataset

def convert_str_to_obj(example):
    for col in example.keys():
        if col.startswith("dialog") or col.startswith("implicit_feedback") or col in ["input_chat_messages", "info"]:
            try:
                example[col] = ast.literal_eval(example[col])
            except (ValueError, SyntaxError):
                try:
                    example[col] = json.loads(example[col])
                except:
                    pass
    if "Locomo" in example["dataset_name"]:
        if example["info"]["category"] == 5:
            example["info"]["golden_answer"] = json.dumps(example["info"]["golden_answer"])
        else:
            example["info"]["golden_answer"] = str(example["info"]["golden_answer"])
    return example

def get_model_response(input_data):
    """
    TODO: Implement this function to call your model.
    
    Args:
        input_data: A dictionary containing the dataset item.
        
    Returns:
        str: The response from your model.
    """
    # Example:
    # prompt = input_data['input_chat_messages']
    # response = my_model.generate(prompt)
    # return response
    
    return "This is a dummy response. Please implement get_model_response."

def main():
    # Configuration
    dataset_name = "NFCats" # Or other datasets in MemoryBench
    output_dir = "results/my_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    dataset = datasets.load_dataset("THUIR/MemoryBench", dataset_name)
    dataset = dataset.map(convert_str_to_obj)
    
    predicts = []
    
    print("Generating predictions...")
    for i, item in tqdm(enumerate(dataset["train"])): # Assuming 'train' split
        response = get_model_response(item)
        
        predicts.append({
            "dataset": dataset_name,
            "test_idx": i, # Using index as test_idx
            "response": response
        })
        
    # Save predictions
    predict_file = os.path.join(output_dir, "predict.json")
    with open(predict_file, "w") as f:
        json.dump(predicts, f, indent=4)
    print(f"Predictions saved to {predict_file}")
    
    # Run Evaluation
    print("Running evaluation...")
    
    # We need to get the dataset object for evaluation
    # This requires the config file. Assuming standard config location.
    # Note: You might need to adjust the config path.
    config_path = "MemoryBench/configs/datasets/domain.json" # Check where NFCats is defined
    
    # Helper to find config
    if dataset_name == "NFCats":
         # NFCats is likely in domain.json or task.json. 
         # Based on file structure, let's try to load it using get_single_dataset if possible,
         # or just rely on evaluate_and_summary if we can pass the dataset object.
         pass

    # Actually, evaluate_and_summary takes a list of dataset objects.
    # Let's try to load the dataset object using the utility.
    try:
        # We need to find where NFCats is defined.
        # Let's assume it's in configs/datasets/domain.json for now, or check other configs.
        # If we can't find it easily, we might need to manually instantiate the evaluator.
        
        # For simplicity, let's try to instantiate the dataset class directly if we can import it.
        from src.dataset.NFCats import NFCats_Dataset
        eval_dataset = NFCats_Dataset() 
        
        # Run evaluation
        results = eval_dataset.evaluate(predicts)
        
        # Save results
        results_file = os.path.join(output_dir, "evaluate_details.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation finished. Results saved to {results_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Please check the configuration and environment variables.")

if __name__ == "__main__":
    main()
