# UI Design Guide — MediFed Web Application

## Design Direction

Clinical, calm and data-first. The interface exists to communicate diagnostic results and training metrics clearly. Every visual decision serves legibility and trust, not decoration.

---

## Theme

Light theme only. The application runs offline on a local machine; no CDN, no web fonts.

---

## Design Tokens

### Colour

| Token | Value | Use |
|---|---|---|
| Background | `#F5F7F8` | Page background |
| Surface | `#FFFFFF` | Cards, panels, table rows |
| Ink | `#1B2A33` | Primary text |
| Muted text | `#5B6B75` | Labels, captions, secondary info |
| Border | `#D8E0E4` | Dividers, input outlines, table rules |
| Primary (deep teal) | `#0F6B6E` | Buttons, active states, key data |
| Caution (amber) | `#C77D0A` | Warnings, moderate confidence |
| Danger | `#B3261E` | Errors, low confidence, critical alerts |

### Hospital Identity Colours

| Hospital | Colour |
|---|---|
| Hospital 1 | `#0F6B6E` |
| Hospital 2 | `#3F5E9A` |
| Hospital 3 | `#A3532B` |

These colours are used only for chart series and hospital labels, never as background fills.

### Typography

System font stack only (the app runs offline):

```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
```

Tabular numerals for all metric values and table cells:

```css
font-variant-numeric: tabular-nums;
```

### Spacing

8 px base grid. All margins, paddings and gaps are multiples of 8 px.

### Borders and Elevation

Thin 1 px borders (`#D8E0E4`) instead of box shadows. No drop shadows, no blur effects.

---

## What Is NOT Allowed

- Gradients on any surface or button.
- Glass / frosted effects.
- Emoji used as icons (use text labels or simple SVG icons).
- Stock illustrations or decorative imagery.
- Purple or blue neon colour palettes.
- Centered hero banners.

---

## Charts

- Every axis must be labelled with a unit where applicable.
- Use colour-blind-safe palettes (the hospital identity colours above are chosen to be distinguishable).
- Prefer direct labels on lines over a separate legend when space allows.
- Grid lines: light (`#D8E0E4`), horizontal only.
- No 3-D effects, no pie charts.

---

## Heatmap Overlay (Grad-CAM)

- Display a colour scale legend beside the heatmap (cool → warm = low → high activation).
- Provide an opacity slider so the clinician can see the underlying X-ray clearly.
- Default opacity: 50 %.
- The heatmap is shown only after a prediction is made; it is never shown on an empty state.

---

## Research Prototype Notice

Every page must display the following notice in a clearly visible location (top of page or persistent footer):

> **Research prototype — not for clinical use.**

Use `color: #B3261E` (danger) and `font-weight: 600`. Do not hide or collapse it.

---

## Page Layouts

### Dashboard

```
┌─────────────────────────────────────────────────────┐
│  MediFed Dashboard          [Research prototype notice] │
├──────────────┬──────────────┬──────────────────────────┤
│  Hospital 1  │  Hospital 2  │  Hospital 3              │
│  Loss curve  │  Loss curve  │  Loss curve              │
│  Acc curve   │  Acc curve   │  Acc curve               │
├──────────────┴──────────────┴──────────────────────────┤
│  Comparison table: FedAvg vs FedBN per hospital        │
│  Comparison chart (bar or line, hospital on x-axis)    │
└─────────────────────────────────────────────────────────┘
```

- Each hospital panel uses its identity colour for chart series only.
- Metrics table uses tabular numerals, 1 px borders, no zebra striping (use border rows instead).

### Diagnosis Page

```
┌─────────────────────────────────────────────────────┐
│  MediFed Diagnosis          [Research prototype notice] │
├─────────────────────────┬───────────────────────────┤
│  Upload X-ray           │  Prediction results        │
│  [Choose file] [Submit] │  1. PNEUMONIA  87 %        │
│                         │  2. NORMAL     11 %        │
│                         │  3. COVID19     2 %        │
│                         │  4. TUBERCULOSIS 0 %       │
├─────────────────────────┴───────────────────────────┤
│  Grad-CAM heatmap overlay                           │
│  [Opacity slider ──────────────── 50 %]             │
│  [X-ray with heatmap overlay]   [Colour scale]      │
└─────────────────────────────────────────────────────┘
```

- Confidence bars use the primary colour for the top prediction, muted for others.
- The heatmap panel is hidden until a prediction is available.

---

## Accessibility

- Colour contrast: all text on its background must meet WCAG AA (4.5 : 1 for normal text, 3 : 1 for large text).
- Keyboard focus: all interactive elements have a visible focus ring (`outline: 2px solid #0F6B6E`).
- Form inputs have associated `<label>` elements.
- Images (X-ray, heatmap) have descriptive `alt` text.
- The research prototype notice is not hidden from screen readers.
