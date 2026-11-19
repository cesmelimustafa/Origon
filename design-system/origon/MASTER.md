# Origon Institutional M&A Analytics Platform — Design System Specification

## 1. Brand Identity and Visual Direction
- **Platform Identity:** Origon Institutional Mergers and Acquisitions Analytics Terminal
- **Aesthetic Direction:** High-density institutional financial workstation engineered for M&A origination analysts. Built around crisp 1px structural gridlines, deep carbon slate surfaces, sovereign champagne gold accents, and cobalt analytical series.
- **Emblem Geometry:** Dual interlocking ribbons converging around a central core node and ascending toward a unified apex target node.

## 2. Color Tokens

| Token | Hex Code | Functional Role |
| :--- | :--- | :--- |
| `--bg-canvas` | `#090C10` | Primary application canvas background |
| `--bg-surface` | `#0F141C` | Command header, filter rail, and card containers |
| `--bg-elevated` | `#151C26` | Sticky table headers, secondary controls, and elevated cells |
| `--bg-hover` | `#1C2533` | Interactive row and button hover surface |
| `--border-subtle` | `#1F2937` | 1px structural grid and table cell dividers |
| `--border-strong` | `#2E3C50` | Active control and input border |
| `--brand-gold` | `#D4AF37` | Origon primary champagne gold accent and active view indicator |
| `--brand-cobalt` | `#3B82F6` | Origon secondary steel blue analytical accent |
| `--signal-positive` | `#10B981` | Positive SHAP M&A drivers and regional valuation discount |
| `--signal-negative` | `#F43F5E` | Negative SHAP M&A deterrents and elevated leverage |
| `--text-primary` | `#F1F5F9` | Primary headings, company titles, and key numerals |
| `--text-secondary` | `#94A3B8` | Column headers and secondary labels |
| `--text-muted` | `#64748B` | Captions, units, and timestamps |

## 3. Typography and Content Discipline
- **UI and Headings:** `Inter` (400, 500, 600, 700).
- **Financial Tables and Numerals:** `JetBrains Mono` (400, 500, 600, 700) with mandatory `font-variant-numeric: tabular-nums` for vertical decimal alignment across all financial columns.
- **Bilingual Support:** Instantaneous `TR | EN` language switcher in the top command bar localizing all UI labels, financial ratios, regional peer benchmarks, and SHAP deal rationales.
- **Content Standards:** Zero emojis, zero decorative quotation marks, and direct institutional financial terminology across both Turkish and English modes.
