from langchain.prompts import ChatPromptTemplate

SYSTEM = """You are a precise assistant. Use only the provided context.
If context is insufficient, say so and ask a follow-up.
Cite briefly like [source:page]."""

USER = """Question: {question}

Context:
{context}

Answer:"""

def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("user", USER),
    ])
