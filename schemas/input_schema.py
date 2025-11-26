# File describing the input type for the Transformer+Lora model we are using
# It will have input, meta, references
class Input:
    def __init__(self, input: str, meta: dict, references: list[str]):
        self.input = input
        self.meta = meta
        self.references = references
    
    def __str__(self):
        return f"Input: {self.input}\nMeta: {self.meta}\nReferences: {self.references}"
    
    def __repr__(self):
        return self.__str__()
    
    input: str
    meta: dict
    references: list[str]    
    
