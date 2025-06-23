---

# Agents.md

## Guidelines for Automated Agents Contributing to This Codebase

This document defines mandatory standards for any automated agent (e.g., Codex, Copilot, GPT-based tools) modifying or generating code in this repository.

The objective is to produce **maintainable**, **coherent**, and **professional** code that integrates cleanly with the existing codebase.

---

### 1. Code Quality and Naming

* Use clear, descriptive, and professional variable and function names.

  * Avoid ambiguous or lazy names (e.g., `tmp`, `foo`, `data1`, etc.).
  * Never generate names that sound careless or unintelligent.
* Avoid magic constants. Use named constants or enumerations.
* Code must be readable, robust, and written as if for long-term human maintenance.

---

### 2. Design Principles

* **Explicit beats implicit**: Be clear in logic, typing, and control flow.
* **Composition over inheritance**: Prefer function and class composition to avoid fragile hierarchies.
* **High cohesion, low coupling**: Group related logic tightly. Minimize interdependencies between unrelated components.
* Eliminate unnecessary duplication. Follow DRY (Don't Repeat Yourself).
* Avoid premature generalization. Abstract only when justified by clear reuse.

---

### 3. Python-Specific Requirements

* Type hints are **mandatory**. Follow standards from Python 3.13 and newer.
* Use `dataclasses` or `namedtuples` to structure related data clearly.
* Avoid legacy or implicit patterns (e.g., duck typing without validation).
* Follow [PEP 8](https://peps.python.org/pep-0008/) and [PEP 484](https://peps.python.org/pep-0484/) consistently.

---

### 4. Structure and Style

* Respect the existing project layout and file boundaries.
* Functions should be focused, short, and do one thing well.
* Minimize unnecessary indirection or abstraction.
* Comment only when necessary. Prioritize explaining *why*, not *what*.

---

### 5. Robustness and Testing

* All meaningful logic must be testable.
* Never silently suppress exceptions. Validate assumptions explicitly.
* Handle edge cases deliberately.
* Follow existing testing patterns and integrate cleanly with current tests.

---

### 6. Documentation

* Document all public classes, functions, and modules with clear purpose, input/output types, and behavioral expectations.
* Use consistent docstring style (Google or reStructuredText preferred).

---

### 7. Agent-Specific Conduct

* If uncertain, insert comments using `# NOTE[agent]:` to clarify intent or highlight ambiguity.
* Do not introduce speculative changes or vague TODOs.
* Preserve intended behavior unless explicitly instructed otherwise.
* Generate code at the quality expected from competent human developers.

---

**Maintainability, clarity, and professionalism are required. No exceptions.**

---
