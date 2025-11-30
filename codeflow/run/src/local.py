from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class ChatModel:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 5120,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.8,
        use_ray: bool = False,
    ):
        """
        Initialize the model and tokenizer
        :param model_path: model path
        :param tensor_parallel_size: GPU parallel number, default is 1
        :param max_model_len: max sequence length for vLLM
        :param dtype: vLLM dtype (e.g., auto, float16, bfloat16, float32)
        :param gpu_memory_utilization: fraction of GPU memory to use
        :param use_ray: whether to use Ray backend (default: False uses multiprocessing)
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_ray = use_ray
        self.llm, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Loading the model and tokenizer
        """
        # Initialize the vLLM's LLM
        backend = "ray" if self.use_ray else "mp"
        llm = LLM(
            model=self.model_path,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            distributed_executor_backend=backend,
        )

        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        return llm, tokenizer

    def format_chat(self, messages):
        """
        Format the message list into the input format required by the model
        :param messages: message list, format is [{'role': 'user', 'content': '...'}, ...]
        :return: formatted prompt
        """
        # 使用 tokenizer 的 apply_chat_template 方法自动格式化
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt

    def generate(self, messages, max_tokens=10000, temperature=0.6, top_p=1):
        """
        Generate model responses
        :param messages: message list, format is [{'role': 'user', 'content': '...'}, ...]
        :param max_tokens: maximum number of tokens generated
        :param temperature: randomness of generation
        :param top_p: sample probability cutoff
        :return: model generation results
        """
        # Set sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Format the message list into the input format required by the model
        prompt = self.format_chat(messages)
        #print(prompt)

        # Making inferences
        output = self.llm.generate(prompt, sampling_params)

        # Returns the generated result
        return output
