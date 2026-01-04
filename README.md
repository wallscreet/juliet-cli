# Juliet Framework and CLI

## **Junctive Unstructured Learning for Incrementally Evolving Transformers**

Juliet is a modular AI framework where independent isomorphic agents develop persistent, personalized intelligence via swappable adapters that inject structured, tagged context into a standardized prompt pipeline enabling plug-and-play, divergent learning without rigid orchestration.

Each *iso* maintains its own **persistent memory** (via vector stores), **facts**, and **task state**, while continuously ingesting unstructured data from conversations, documents (`.pdf`, `.txt`, etc.), and other sources. Over time, isos fine-tune **lightweight adapters** against their unique experiential history, giving rise to distinct traits, behaviors, and emergent intelligence despite originating from a shared structural blueprint.

Juliet goes beyond conventional RAG pipelines. Rather than treating retrieval as a static augmentation step, the framework models learning as a dynamic process where **memory**, **state-space evolution**, and **contextual perturbations** interact and produce adaptive, creative, and increasingly individualized agents.

---

## Project-level management with sandboxed workspaces

---

## Dynamic Context System

Juliet’s core is a **dynamic, adapter-driven context pipeline** designed for composability and long-term evolution.

* Each context source (facts, memory, knowledge, history, etc.) is implemented as an independent adapter.
* Adapters output a `List[dict]` of standardized OpenAI-compatible messages (`system`, `user`, `assistant` only).
* The prompt pipeline is assembled deterministically:

  1. System instructions
  2. Adapter outputs (stacked in configurable order)
  3. User request
  4. Assistant prefix for forced continuation

```python
# Pipeline assembly example
messages = []

# System instructions
messages.extend(system_adapter.build_messages())

# Stack adapters in order
messages.extend(facts_adapter.build_messages())
messages.extend(memory_adapter.build_messages())
messages.extend(knowledge_adapter.build_messages())
messages.extend(history_adapter.build_messages())

# User request
messages.append({"role": "user", "content": f"<user>{user_request}</user>"})

# Forced assistant prefix
messages.append({"role": "assistant", "content": "<assistant>"})

# Final prompt ready for LLM
```

To preserve semantic boundaries and guide model reasoning, adapter content is compartmentalized using XML-like tags (e.g. `<facts>`, `<memory>`, `<knowledge>`). For example:

```xml
<knowledge>
  <sherlock_holmes>
    Relevant passages from Sherlock Holmes...
  </sherlock_holmes>
  <atlas_shrugged>
    Relevant passages from Ayn Rand...
  </atlas_shrugged>
</knowledge>
```

**Summary:** Highly modular, swappable context injection with a strict message format and tagged structure optimized for clarity, control, and emergent learning.

---

## TUI Interface for a Dynamic Learning Framework

Juliet includes a terminal-based interface designed to make agent evolution observable and hands-on:

* Agent-centric contextual views
* Modularity-first, plug-and-play adapter architecture
* Feed-forward prompt construction
* Recursive tool calling
* Model workspace with full CRUD file access
* ChromaDB-backed vector memory
* YAML-driven instructions, configuration, and history

![alt text](framework.png)
