"""
plot_grc_flow.py
================
Generates a publication-quality figure illustrating the flow of the
Generative Reconstruction Compression (GRC) pipeline.

Two lanes:
  SENDER  — compress the final program
  RECEIVER — reconstruct the final program

Saved to: results/generative_reconstruction_compression/grc_flow.png
"""

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import sys
import matplotlib
matplotlib.use("Agg")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
OUT_DIR = os.path.join(
    PROJECT_ROOT, "results/generative_reconstruction_compression")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "grc_flow.png")

# ── Colour palette ─────────────────────────────────────────────────────────────
BG = "#0f1117"
LANE_BG = "#1a1d27"
LANE_EDGE = "#2e3250"

C_PROMPT = "#4f8ef7"   # blue  – prompt / seed
C_GEN = "#a78bfa"   # purple – LLM generation
C_EDIT = "#f59e0b"   # amber  – developer edits
C_DIFF = "#34d399"   # green  – diff / opcodes
C_COMP = "#f472b6"   # pink   – compression
C_RECON = "#38bdf8"   # sky    – reconstruction
C_VERIFY = "#4ade80"   # lime   – verification
C_ARROW = "#94a3b8"   # slate  – arrows
C_XMIT = "#fb923c"   # orange – transmitted bits

TEXT_MAIN = "#f1f5f9"
TEXT_SUB = "#94a3b8"
TEXT_LABEL = "#cbd5e1"

FONT_MAIN = "DejaVu Sans"

# ── Figure layout ──────────────────────────────────────────────────────────────
FIG_W, FIG_H = 18, 10
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor(BG)

# ── Helper functions ───────────────────────────────────────────────────────────


def rounded_rect(ax, x, y, w, h, color, alpha=1.0, lw=1.5, edge=None, zorder=3):
    edge = edge or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=edge,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def node(ax, cx, cy, w, h, label, sublabel=None, color="#4f8ef7", text_color=TEXT_MAIN):
    """Draw a rounded box centred at (cx, cy)."""
    x, y = cx - w/2, cy - h/2
    rounded_rect(ax, x, y, w, h, color=color, alpha=0.18, edge=color, lw=2)
    ax.text(cx, cy + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=text_color, fontfamily=FONT_MAIN, zorder=5)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=TEXT_SUB, fontfamily=FONT_MAIN, zorder=5)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, label=None, lw=1.8,
          connectionstyle="arc3,rad=0.0", zorder=4):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=zorder,
    )
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my + 0.18, label,
                ha="center", va="bottom", fontsize=7.5,
                color=color, fontfamily=FONT_MAIN, zorder=5)


def lane_bg(ax, y_bot, y_top, label, label_color):
    """Draw a horizontal lane background."""
    rounded_rect(ax, 0.3, y_bot, FIG_W - 0.6, y_top - y_bot,
                 color=LANE_BG, alpha=1.0, edge=LANE_EDGE, lw=1.2, zorder=1)
    ax.text(0.65, (y_bot + y_top) / 2, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=label_color, rotation=90, fontfamily=FONT_MAIN, zorder=2)


def bit_badge(ax, cx, cy, label, color=C_XMIT):
    """Small badge showing what is transmitted."""
    w, h = 1.5, 0.38
    x, y = cx - w/2, cy - h/2
    rounded_rect(ax, x, y, w, h, color=color, alpha=0.25,
                 edge=color, lw=1.2, zorder=6)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=7,
            color=color, fontweight="bold", fontfamily=FONT_MAIN, zorder=7)


# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(FIG_W/2, 9.6, "Generative Reconstruction Compression (GRC) — Pipeline Flow",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=TEXT_MAIN, fontfamily=FONT_MAIN)

# ── Lane backgrounds ───────────────────────────────────────────────────────────
#   Sender lane:   y  4.6 – 9.1
#   Receiver lane: y  0.5 – 4.4
lane_bg(ax, 4.65, 9.1,  "SENDER",   "#a78bfa")
lane_bg(ax, 0.45, 4.45, "RECEIVER", "#38bdf8")

# ── Shared transmitted-bits channel ───────────────────────────────────────────
# Drawn as a thin horizontal band between the two lanes
rounded_rect(ax, 0.3, 4.35, FIG_W - 0.6, 0.38,
             color=C_XMIT, alpha=0.08, edge=C_XMIT, lw=1.0, zorder=2)
ax.text(FIG_W/2, 4.54, "◀  Transmitted Bitstream  ▶",
        ha="center", va="center", fontsize=8, color=C_XMIT,
        fontfamily=FONT_MAIN, zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# SENDER lane  (y ≈ 5.0 – 8.8)
# Steps: Prompt+Seed → LLM Generate → Developer Edits → Tokenise → Diff
#        → Compress Prompt (LLMzip) + Compress Opcodes + Compress Content
# ══════════════════════════════════════════════════════════════════════════════

NW, NH = 2.2, 0.80   # node width / height
SY = 7.6             # top row y-centre (sender)
SY2 = 5.8            # bottom row y-centre (sender)

# X positions for sender top row
XS = [1.6, 4.1, 6.6, 9.1, 11.6]

# Top row nodes
node(ax, XS[0], SY, NW, NH, "Prompt + Seed",
     sublabel="user intent + RNG seed", color=C_PROMPT)
node(ax, XS[1], SY, NW, NH, "LLM Generate",
     sublabel="frontier model (API)", color=C_GEN)
node(ax, XS[2], SY, NW, NH, "Developer Edits",
     sublabel="add types, docs, fixes", color=C_EDIT)
node(ax, XS[3], SY, NW, NH, "Tokenise Both",
     sublabel="raw_ids, edited_ids", color=C_DIFF)
node(ax, XS[4], SY, NW, NH, "Diff (Myers /\nRatcliff)",
     sublabel="opcode stream", color=C_DIFF)

# Arrows along top row
for i in range(len(XS) - 1):
    arrow(ax, XS[i] + NW/2, SY, XS[i+1] - NW/2, SY)

# Labels on arrows
arrow_labels = ["", "raw output", "final program", "", ""]
for i, lbl in enumerate(arrow_labels):
    if lbl and i < len(XS)-1:
        mx = (XS[i] + NW/2 + XS[i+1] - NW/2) / 2
        ax.text(mx, SY + 0.32, lbl, ha="center", va="bottom",
                fontsize=7, color=TEXT_SUB, fontfamily=FONT_MAIN)

# Bottom row: three compression outputs
XC = [3.0, 7.5, 12.0]
node(ax, XC[0], SY2, NW, NH, "LLMzip\n(Prompt)",
     sublabel="entropy-coded tokens", color=C_COMP)
node(ax, XC[1], SY2, NW, NH, "Entropy Code\nOpcodes",
     sublabel="tag + length stream", color=C_COMP)
node(ax, XC[2], SY2, NW, NH, "Ctx-Aware\nContent Bits",
     sublabel="insert/replace tokens", color=C_COMP)

# Arrows from Prompt+Seed → LLMzip(Prompt)
arrow(ax, XS[0], SY - NH/2, XC[0], SY2 + NH/2,
      color=C_COMP, connectionstyle="arc3,rad=0.15")
# Arrow from Diff → Entropy Code Opcodes
arrow(ax, XS[4], SY - NH/2, XC[1], SY2 + NH/2,
      color=C_COMP, connectionstyle="arc3,rad=-0.1")
# Arrow from Diff → Ctx-Aware Content
arrow(ax, XS[4] + 0.3, SY - NH/2, XC[2], SY2 + NH/2,
      color=C_COMP, connectionstyle="arc3,rad=0.1")

# Seed badge (32 bits, fixed)
bit_badge(ax, 1.6, SY2, "Seed  32 bits", color=C_PROMPT)

# Downward arrows from compression nodes to transmitted channel
for xc in XC:
    arrow(ax, xc, SY2 - NH/2, xc, 4.73, color=C_XMIT, lw=1.5)

arrow(ax, 1.6, SY2 - 0.19, 1.6, 4.73, color=C_XMIT, lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# RECEIVER lane  (y ≈ 0.7 – 4.2)
# Steps: Decode Prompt → Re-generate (same model+seed) → Apply Diff → Verify
# ══════════════════════════════════════════════════════════════════════════════

RY = 2.5   # main row y-centre (receiver)
RY2 = 1.0   # output row

XR = [1.6, 4.1, 6.6, 9.1, 11.6, 14.1]

node(ax, XR[0], RY, NW, NH, "Decode Prompt\n+ Seed",
     sublabel="LLMzip decompress", color=C_RECON)
node(ax, XR[1], RY, NW, NH, "Re-Generate\nRaw Output",
     sublabel="same model + seed", color=C_GEN)
node(ax, XR[2], RY, NW, NH, "Decode Opcodes",
     sublabel="entropy decode", color=C_RECON)
node(ax, XR[3], RY, NW, NH, "Decode Content\nTokens",
     sublabel="ctx-aware decode", color=C_RECON)
node(ax, XR[4], RY, NW, NH, "Apply Diff\nOpcodes",
     sublabel="equal/insert/replace", color=C_DIFF)
node(ax, XR[5], RY, NW, NH, "Verify\nExact Match",
     sublabel="token-level check", color=C_VERIFY)

# Arrows along receiver row
for i in range(len(XR) - 1):
    arrow(ax, XR[i] + NW/2, RY, XR[i+1] - NW/2, RY)

# Upward arrows from transmitted channel into receiver nodes
for xr, label in [(XR[0], "prompt bits"), (XR[2], "opcode bits"), (XR[3], "content bits")]:
    arrow(ax, xr, 4.73, xr, RY + NH/2, color=C_XMIT, lw=1.5)

# Final output label
node(ax, XR[5] + 2.2, RY, NW, NH, "Final Program\n(Reconstructed)",
     sublabel="exact bit-for-bit match", color=C_VERIFY)
arrow(ax, XR[5] + NW/2, RY, XR[5] + 2.2 - NW/2, RY, color=C_VERIFY)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    (C_PROMPT, "Prompt / Seed"),
    (C_GEN,    "LLM Generation"),
    (C_EDIT,   "Developer Edits"),
    (C_DIFF,   "Diff / Tokenise"),
    (C_COMP,   "Compression"),
    (C_RECON,  "Decompression"),
    (C_VERIFY, "Verification"),
    (C_XMIT,   "Transmitted Bits"),
]
handles = [
    mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.7, label=l)
    for c, l in legend_items
]
leg = ax.legend(
    handles=handles, loc="lower right",
    bbox_to_anchor=(0.995, 0.01),
    ncol=4, fontsize=8,
    facecolor=LANE_BG, edgecolor=LANE_EDGE,
    labelcolor=TEXT_LABEL, framealpha=0.95,
    handlelength=1.2, handleheight=0.9,
)

# ── Bit breakdown annotation ───────────────────────────────────────────────────
annot = (
    "Transmitted:  LLMzip(prompt) + 32-bit seed\n"
    "            + entropy(opcodes) + ctx-aware(content)"
)
# placeholder already drawn above
ax.text(FIG_W/2, 4.54 - 0.0, "", ha="center")

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT_PATH}")
