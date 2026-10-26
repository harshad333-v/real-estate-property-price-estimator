---
name: front-refactor
description: "Safely refactor CSS and vanilla JavaScript in the Django frontend."
---

# Frontend refactoring skill (adapted)

Use this workflow for refactoring the project's CSS or JavaScript without changing observable behavior.

## Workflow
1. Read the entire file and identify its callers, DOM IDs, CSS selectors, API contracts, and event flow.
2. Default to preview mode: propose a structured set of changes and do not edit.
3. Only apply changes when explicitly asked to apply them.
4. Group proposed changes by dead code, naming, simplification, and modernization.
5. Preserve endpoint paths, payload field names, response fields, CSRF headers, form defaults, and accessibility behavior.
6. Avoid introducing dependencies or frameworks unless the user asks for them.
7. After changes, perform syntax checks and manually verify the relevant browser flow where possible.

## Safety requirements
- Do not refactor code without reading the complete target file.
- Avoid behavior-changing transformations.
- Do not remove meaningful comments, validation, error handling, or accessibility attributes.
- Do not use `innerHTML` for untrusted content.
- Explain any change that requires coordinated edits across files.

## Output
Preview mode: structured diff with line references, before/after snippets, and reasons.
Apply mode: concise report of categories changed and any manual verification still required.
