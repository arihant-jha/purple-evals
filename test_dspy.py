import os
import dspy
from typing import Literal

CLASSES = ["cat", "dog"]
trainset = [
    dspy.Example(
        text="Which animal meows?",
        label=CLASSES[0]
    ).with_inputs("text", "label"),
    dspy.Example(
        text="Which animal barks?",
        label=CLASSES[1]
    ).with_inputs("text", "label")
]

lm = dspy.LM("gemini-2.0-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))
dspy.configure(lm=lm)

signature = dspy.Signature("text -> label").with_updated_fields("label", type_=Literal[tuple(CLASSES)])
print(signature)

classify = dspy.ChainOfThought(signature)
print(classify)

optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=2)
print(optimizer)

# optimized = optimizer.compile(classify, trainset=trainset)

predict = dspy.Predict(signature)
print(predict(sentence="Which animal meows?"))
