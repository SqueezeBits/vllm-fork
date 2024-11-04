import sys
import pathlib

a = pathlib.Path(sys.argv[1])
b = pathlib.Path(sys.argv[2])
assert a.exists()
assert b.exists()

for i in range(2, 100):
    logits = f"{i}.logits"
    tokens = f"{i}.tokens"

    a_logit_file = (a / logits).open() 
    b_logit_file = (b / logits).open() 
    a_token_file = (a / tokens).open() 
    b_token_file = (b / tokens).open() 

    a_logits = a_logit_file.readlines()
    b_logits = b_logit_file.readlines()
    a_tokens = a_token_file.readlines()
    b_tokens = b_token_file.readlines()

    if len(a_tokens) != len(b_tokens):
        print(i)
        import pdb; pdb.set_trace()

    if not all(at == bt for at, bt in zip(a_tokens, b_tokens)):
        import pdb; pdb.set_trace()



    
    
