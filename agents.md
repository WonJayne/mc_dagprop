# Agents.md

## 🧠 Guidelines for Automated Agents Editing This Codebase

This document defines coding principles that **must be respected** by automated agents contributing to this repository (e.g., Codex, Copilot, GPT-based assistants).

---

### 1. ✅ Code Quality Principles

* **Write maintainable code**: Solutions should favor **clarity over cleverness** and **consistency over novelty**.
* **Use descriptive and non-embarrassing identifiers**:

  * Variables like `tmp`, `data`, `foo`, `var123`, or `lol` are prohibited unless clearly justified.
  * Prefer `train_departure_time` over `tdt` unless context demands brevity.
* **No magic numbers or strings**: Use constants or enums.

---

### 2. 💡 Design Philosophy

* **Prefer strong coupling, low cohesion**:

  * Related logic should live together.
  * Do not scatter functionality across loosely connected utility files.
* **Minimize duplication**:

  * Reuse abstractions, define shared helpers, respect DRY (Don't Repeat Yourself).
  * Copy-pasting code with minor tweaks is forbidden.

---

### 3. 🧱 Structure and Style

* Respect project structure and file naming conventions.
* Keep functions short (≤ 40 LOC is a good default). Refactor if they grow too large.
* Use type annotations (in Python), const correctness (in C++), etc.
* Comments must explain *why*, not *what*, unless code is non-obvious.

---

### 4. 🧪 Testing and Robustness

* Every non-trivial function must be testable.
* Do not silently catch exceptions or suppress errors without logging or justification.
* Fail loudly when assumptions are violated unless otherwise documented.

---

### 5. 📚 Documentation

* All public functions and classes must have a docstring or comment block.
* Document side effects and expectations (e.g., mutability, I/O).

---

### 6. 🚫 Anti-Patterns (Never Do This)

* Don’t invent DSLs unless absolutely necessary.
* Don’t name everything `manager`, `handler`, or `util`.
* Avoid `print` for debugging—use logging or tracing frameworks.
* Don’t add TODOs you don’t intend to fix.

---

### 7. 🤖 Agent-Specific Rules

* If unsure, **ask** (via prompt or comment) instead of assuming.
* Leave notes (`# NOTE[agent-name]:`) to clarify uncertain decisions.
* If refactoring, preserve behavior unless explicitly told otherwise.

---

**Remember**: Your output will be read and maintained by humans. Write accordingly.

---

