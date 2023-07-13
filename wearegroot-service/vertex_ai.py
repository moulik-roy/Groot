from vertexai.preview.language_models import TextGenerationModel

def f1():
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")

    prompt = """
Provide a very short summary, no more than three sentences, for the following article:

Our quantum computers work by manipulating qubits in an orchestrated fashion that we call quantum algorithms.
The challenge is that qubits are so sensitive that even stray light can cause calculation errors — and the problem worsens as quantum computers grow.
This has significant consequences, since the best quantum algorithms that we know for running useful applications require the error rates of our qubits to be far lower than we have today.
To bridge this gap, we will need quantum error correction.
Quantum error correction protects information by encoding it across multiple physical qubits to form a “logical qubit,” and is believed to be the only way to produce a large-scale quantum computer with error rates low enough for useful calculations.
Instead of computing on the individual qubits themselves, we will then compute on logical qubits. By encoding larger numbers of physical qubits on our quantum processor into one logical qubit, we hope to reduce the error rates to enable useful quantum algorithms.

Summary:

"""
    
    response = generation_model.predict(prompt, temperature=0.2, max_output_tokens=1024, top_k=40, top_p=0.8).text

    return response
