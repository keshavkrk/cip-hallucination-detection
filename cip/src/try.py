from preprocessing.module2_preprocess import module2_process

out = module2_process("Who invented the telephone?")

print(out["answer"])
print(out["input_ids"].shape)
