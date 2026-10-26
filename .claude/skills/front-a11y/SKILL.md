---
name: front-a11y
description: "Audit or fix accessibility issues in the project's Django HTML templates."
---

# Frontend accessibility skill (adapted)

Use this checklist when reviewing or changing HTML templates, CSS, and vanilla JavaScript in this repository.

## Workflow
1. Read the complete target template and any JavaScript handling its forms.
2. In audit mode, report findings by Critical, Major, Minor severity without edits.
3. In fix mode, apply safe fixes while preserving Django routes, API field names, request method, CSRF handling, and existing behavior.
4. Re-check keyboard access, form labels, error messaging, semantics, focus indicators, responsive behavior, and reduced-motion preferences.

## Required checks
- Set document language and responsive viewport.
- Use semantic landmarks and headings in a logical hierarchy.
- Every input/select/textarea has a visible, programmatically associated label.
- Required fields are indicated both visually and with native validation.
- Buttons have clear accessible names and visible keyboard focus.
- Dynamic status and errors use an appropriate live region; errors should not be conveyed by color alone.
- Hidden content remains hidden; do not use CSS that unintentionally overrides the `hidden` attribute.
- Provide sufficient contrast, usable touch targets, and responsive layouts.
- Honor `prefers-reduced-motion`.
- Do not add ARIA where native HTML semantics already provide the behavior.
- Keep user data safe: don't log or send form values to unrelated services.
- Preserve CSRF protection and avoid inserting server/user-controlled text with `innerHTML`; prefer `textContent`.

## Output
For an audit, return findings grouped by severity with location, issue, and recommended fix. For fixes, summarize what changed and what needs manual verification.
