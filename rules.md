# Project AI Rules & Coding Standards

## 1. Architectural Modularity
- **Max Lines Per File:** Strict limit of < 400 lines per file to ensure modularity.
- **Self-Contained:** Each module must perform a single, well-defined responsibility.

## 2. Medical AI Data Integrity
- **Zero Data Leakage:** Strict enforcement of zero data leakage between Train, Validation, and Test splits.
- **Augmentations:** Applied ONLY to the Training set. Validation and Test sets must remain untouched except for normalization and resizing.

## 3. Device & Resource Management
- **Dynamic Allocation:** Models and tensors must dynamically allocate to the available device (`cuda` or `cpu`).
- **Memory Cleanup:** Use `torch.cuda.empty_cache()` hooks where appropriate to prevent Out-Of-Memory (OOM) errors, especially in Colab environments.

## 4. Coding Standards
- **PEP8 Compliance:** Code must adhere to PEP8 formatting guidelines.
- **Type Annotations:** Strict usage of Python's `typing` module for all function signatures and class definitions.
- **Logging:** Use Python's structured `logging` module. Raw `print()` statements are prohibited in production code (except for explicit standalone test scripts).

## 5. Error Handling & Fallbacks
- **Graceful Failures:** Implement robust exception handling for missing, corrupt, or unreadable dataset files.
- **Edge Cases:** Ensure mathematical stability (e.g., epsilon smoothing for division by zero) and graceful fallbacks (e.g., empty segmentation masks in Hausdorff distance calculations).
