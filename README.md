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
class ContextPipeline:
    def __init__(self, chroma_store: ChromaMemoryStore, message_cache: MessageCacheAdapter, iso_name: str, user_name: str):
        
        self.instructions = ModelInstructions(method="load", assistant_name=iso_name)
        self.adapters = OrderedDict()
        self.chroma_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/chroma_store"
        self.semantic_memory_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/semantic_memory.json"
        self.episodic_memory_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/episodic_memory.json"
        self.facts_memory_path = f"isos/{iso_name.lower().strip()}/users/{user_name.lower().strip()}/facts_memory.json"

        # === Register Context Adapters === #
        self.register_adapter("timestamp", TimestampAdapter())
        self.register_adapter("semantic", SemanticMemoryAdapter(chroma_store=chroma_store))
        self.register_adapter("episodic", EpisodicMemoryAdapter(chroma_store=chroma_store))
        # TODO: self.register_adapter("procedural", ProceduralMemoryAdapter(chroma_store=chroma_store))
        # TODO: self.register_adapter("facts", FactStoreAdapter(chroma_store=chroma_store))
        self.register_adapter("message_cache", message_cache)
        self.register_adapter("user_request", UserRequestAdapter(tag_name="user"))
        self.register_adapter("assistant_prefix", AssistantPrefixAdapter(prefix="<assistant>"))

    def register_adapter(self, name: str, adapter: BaseContextAdapter):
        self.adapters[name] = adapter

    def build_messages(self, user_request: str) -> List[Dict[str, str]]:
        """
        Build the full message list by combining fixed instructions and all adapters.
        Adapters that require the current user query receive it; others do not.
        """
        messages = [
            {"role": "system", "content": f"<system>{self.instructions.system_message}</system>"},
            {"role": "assistant", "content": f"<assistant_intro>{self.instructions.assistant_intro}</assistant_intro>"},
            {"role": "system", "content": f"<focus>{self.instructions.assistant_focus}</focus>"},
        ]

        for adapter in self.adapters.values():
            if adapter.requires_user_request:
                msgs = adapter.build_messages(user_request=user_request)
            else:
                msgs = adapter.build_messages()
            if msgs:
                messages.extend(msgs)

        return messages
```

To preserve semantic boundaries and guide model reasoning, adapter content is compartmentalized using XML-like tags (e.g. `<facts>`, `<memory>`, `<knowledge>`). For example:

```xml
<semantic>
  <The_Adventures_Of_Sherlock_Holmes>
    Relevant passages from The Adventures of Sherlock Holmes...
  </The_Adventures_Of_Sherlock_Holmes>
  <Attention_Is_All_You_Need>
    Relevant passages from the Attention is All You Need paper...
  </Attention_Is_All_You_Need>
</semantic>
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
