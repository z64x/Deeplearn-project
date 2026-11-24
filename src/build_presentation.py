"""Generate a 14-slide 16:9 PowerPoint presentation for Project 2.6: Gas Sensor Array Drift."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PPTX = PROJECT_ROOT / "report" / "Gas_Sensor_Array_Drift_Presentation.pptx"

SLIDE_W = 13.333
SLIDE_H = 7.500
FONT_NAME = "Segoe UI"
TOTAL_SLIDES = 14

# Theme Palette ("Deep Slate & Teal Scientific")
C_DARK_BG = RGBColor(0x0F, 0x17, 0x2A)
C_DARK_CARD = RGBColor(0x1E, 0x29, 0x3B)
C_DARK_BORDER = RGBColor(0x33, 0x41, 0x55)
C_TEXT_LIGHT = RGBColor(0xF8, 0xFA, 0xFC)
C_TEXT_MUTED_DARK = RGBColor(0x94, 0xA3, 0xB8)
C_TEAL_BRIGHT = RGBColor(0x14, 0xB8, 0xA6)
C_BLUE_BRIGHT = RGBColor(0x38, 0xBD, 0xF8)

C_LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_BORDER = RGBColor(0xCB, 0xD5, 0xE1)
C_DIVIDER = RGBColor(0xE2, 0xE8, 0xF0)
C_TEXT_DARK = RGBColor(0x0F, 0x17, 0x2A)
C_TEXT_SEC = RGBColor(0x47, 0x55, 0x69)
C_TEXT_FOOTER = RGBColor(0x64, 0x74, 0x8B)
C_PILL_TEXT = RGBColor(0x33, 0x41, 0x55)

C_TEAL = RGBColor(0x0D, 0x94, 0x88)
C_TEAL_TINT = RGBColor(0xF0, 0xFD, 0xFA)
C_BLUE = RGBColor(0x25, 0x63, 0xEB)
C_BLUE_TINT = RGBColor(0xEF, 0xF6, 0xFF)
C_AMBER = RGBColor(0xD9, 0x77, 0x06)
C_AMBER_TINT = RGBColor(0xFE, 0xF3, 0xC7)
C_CORAL = RGBColor(0xDC, 0x26, 0x26)
C_CORAL_TINT = RGBColor(0xFE, 0xF2, 0xF2)
C_EMERALD = RGBColor(0x05, 0x96, 0x69)
C_EMERALD_TINT = RGBColor(0xEC, 0xFD, 0xF5)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def set_slide_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill_color: RGBColor | None = None,
    border_color: RGBColor | None = None,
    border_pt: float = 1.0,
    rounded: bool = False,
):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    if rounded:
        try:
            shape.adjustments[0] = 0.04
        except Exception:
            pass
    if fill_color is not None:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()

    if border_color is not None:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    font_size: float = 11.0,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = C_TEXT_DARK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    margin_lr: float = 0.06,
    margin_tb: float = 0.04,
):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin_lr)
    tf.margin_right = Inches(margin_lr)
    tf.margin_top = Inches(margin_tb)
    tf.margin_bottom = Inches(margin_tb)
    if text:
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.name = FONT_NAME
        r.font.size = Pt(font_size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def add_bullets(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    items: list[tuple[str, str]],
    font_size: float = 11.0,
    lead_color: RGBColor = C_TEXT_DARK,
    body_color: RGBColor = C_TEXT_SEC,
    space_after: float = 6.0,
    bullet_prefix: str = "• ",
):
    """Add rich bullet paragraphs with bold lead phrase + regular explanation."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)

    for idx, (lead, rest) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.15

        if bullet_prefix:
            r0 = p.add_run()
            r0.text = bullet_prefix
            r0.font.name = FONT_NAME
            r0.font.size = Pt(font_size)
            r0.font.bold = True
            r0.font.color.rgb = C_TEAL if lead_color == C_TEXT_DARK else C_TEAL_BRIGHT

        if lead:
            r1 = p.add_run()
            r1.text = lead
            r1.font.name = FONT_NAME
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = lead_color

        if rest:
            r2 = p.add_run()
            r2.text = rest
            r2.font.name = FONT_NAME
            r2.font.size = Pt(font_size)
            r2.font.bold = False
            r2.font.color.rgb = body_color
    return tb


def set_notes(slide, notes_text: str) -> None:
    slide.notes_slide.notes_text_frame.text = notes_text


def add_header_footer(slide, slide_num: int, eyebrow: str, title: str) -> None:
    set_slide_bg(slide, C_LIGHT_BG)
    # Top accent bar
    add_rect(slide, 0.0, 0.0, SLIDE_W, 0.07, fill_color=C_TEAL)
    # Eyebrow
    add_textbox(
        slide,
        0.60,
        0.22,
        11.10,
        0.24,
        text=eyebrow.upper(),
        font_size=9.5,
        bold=True,
        color=C_TEAL,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    # Action title
    add_textbox(
        slide,
        0.60,
        0.46,
        11.10,
        0.46,
        text=title,
        font_size=21.0,
        bold=True,
        color=C_TEXT_DARK,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    # Slide counter pill
    pill = add_rect(
        slide, 11.95, 0.34, 0.78, 0.30, fill_color=C_DIVIDER, rounded=True
    )
    pill.adjustments[0] = 0.35
    add_textbox(
        slide,
        11.95,
        0.34,
        0.78,
        0.30,
        text=f"{slide_num:02d} / {TOTAL_SLIDES}",
        font_size=9.5,
        bold=True,
        color=C_PILL_TEXT,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    # Header divider
    add_rect(slide, 0.60, 1.02, 12.133, 0.015, fill_color=C_DIVIDER)
    # Footer
    add_textbox(
        slide,
        0.60,
        7.16,
        12.133,
        0.24,
        text="Project 2.6 · Gas Sensor Array Drift · Deep Learning Course Workshop",
        font_size=9.0,
        color=C_TEXT_FOOTER,
        margin_lr=0.0,
        margin_tb=0.0,
    )


def add_fitted_picture(
    slide,
    img_path: Path,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
    pad: float = 0.14,
    card_bg: RGBColor = C_WHITE,
    card_border: RGBColor = C_BORDER,
) -> None:
    """Place an image inside an elevated white card, preserving exact aspect ratio."""
    add_rect(
        slide,
        box_x,
        box_y,
        box_w,
        box_h,
        fill_color=card_bg,
        border_color=card_border,
        border_pt=1.0,
        rounded=True,
    )
    with Image.open(img_path) as im:
        px_w, px_h = im.size
    img_ratio = px_w / px_h

    inner_w = box_w - 2 * pad
    inner_h = box_h - 2 * pad
    box_ratio = inner_w / inner_h

    if img_ratio > box_ratio:
        draw_w = inner_w
        draw_h = inner_w / img_ratio
    else:
        draw_h = inner_h
        draw_w = inner_h * img_ratio

    draw_x = box_x + (box_w - draw_w) / 2.0
    draw_y = box_y + (box_h - draw_h) / 2.0
    slide.shapes.add_picture(
        str(img_path), Inches(draw_x), Inches(draw_y), Inches(draw_w), Inches(draw_h)
    )


def add_kpi_card(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    label: str,
    sub: str = "",
    accent: RGBColor = C_TEAL,
    bg: RGBColor = C_WHITE,
    border: RGBColor = C_BORDER,
    dark: bool = False,
) -> None:
    add_rect(
        slide, x, y, w, h, fill_color=bg, border_color=border, border_pt=1.0, rounded=True
    )
    add_rect(slide, x, y, w, 0.06, fill_color=accent)
    val_color = accent if dark else accent
    lbl_color = C_TEXT_LIGHT if dark else C_TEXT_DARK
    sub_color = C_TEXT_MUTED_DARK if dark else C_TEXT_SEC

    add_textbox(
        slide,
        x + 0.14,
        y + 0.14,
        w - 0.28,
        0.42,
        text=value,
        font_size=20.0,
        bold=True,
        color=val_color,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_textbox(
        slide,
        x + 0.14,
        y + 0.56,
        w - 0.28,
        0.28,
        text=label,
        font_size=11.0,
        bold=True,
        color=lbl_color,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    if sub:
        add_textbox(
            slide,
            x + 0.14,
            y + 0.84,
            w - 0.28,
            0.24,
            text=sub,
            font_size=9.5,
            color=sub_color,
            margin_lr=0.0,
            margin_tb=0.0,
        )


def add_flow_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    fill_color: RGBColor,
    border_color: RGBColor,
    title_color: RGBColor = C_TEXT_DARK,
) -> None:
    add_rect(
        slide,
        x,
        y,
        w,
        h,
        fill_color=fill_color,
        border_color=border_color,
        border_pt=1.0,
        rounded=True,
    )
    add_textbox(
        slide,
        x + 0.12,
        y + 0.06,
        w - 0.24,
        0.26,
        text=title,
        font_size=11.0,
        bold=True,
        color=title_color,
        align=PP_ALIGN.CENTER,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_textbox(
        slide,
        x + 0.12,
        y + 0.30,
        w - 0.24,
        0.24,
        text=subtitle,
        font_size=9.5,
        color=C_TEXT_SEC,
        align=PP_ALIGN.CENTER,
        margin_lr=0.0,
        margin_tb=0.0,
    )


# ---------------------------------------------------------------------------
# Slide Builders
# ---------------------------------------------------------------------------
def build_slide_01_title(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_DARK_BG)

    # Left vertical accent bar
    add_rect(slide, 0.0, 0.0, 0.14, SLIDE_H, fill_color=C_TEAL_BRIGHT)

    # Eyebrow badge
    add_rect(
        slide,
        0.80,
        1.10,
        3.95,
        0.34,
        fill_color=C_DARK_CARD,
        border_color=C_TEAL_BRIGHT,
        border_pt=1.0,
        rounded=True,
    )
    add_textbox(
        slide,
        0.80,
        1.10,
        3.95,
        0.34,
        text="DEEP LEARNING PROJECT · SECTION 2.6",
        font_size=10.0,
        bold=True,
        color=C_TEAL_BRIGHT,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    # Main Title
    add_textbox(
        slide,
        0.80,
        1.68,
        11.60,
        1.45,
        text="Multi-Class Gas Classification Under\nTemporal Sensor Array Drift",
        font_size=34.0,
        bold=True,
        color=C_TEXT_LIGHT,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    # Subtitle
    add_textbox(
        slide,
        0.80,
        3.22,
        11.30,
        0.72,
        text=(
            "Comparative evaluation of GasMLP, 1D-CNN, and UMAP manifold embeddings "
            "under Random Stratified CV vs. Chronological Expanding-Window CV across 36 months."
        ),
        font_size=14.0,
        color=C_TEXT_MUTED_DARK,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    # 4 Glassmorphic KPI cards
    kpis = [
        ("13,910", "MOX Measurements", "0 NaNs · 0 Infs · 0 Duplicates", C_TEAL_BRIGHT),
        ("128 (16×8)", "Sensor Feature Grid", "8 descriptors × 16 chemical sensors", C_BLUE_BRIGHT),
        ("6 Gases", "10 Chronological Batches", "36-month sensor aging horizon", C_TEAL_BRIGHT),
        ("PyTorch + CUDA", "Reproducible Pipeline", "RTX 3050 GPU · Fixed Seed 42", C_BLUE_BRIGHT),
    ]
    card_w = 2.78
    gap = 0.24
    start_x = 0.80
    for idx, (val, lbl, sub, acc) in enumerate(kpis):
        add_kpi_card(
            slide,
            start_x + idx * (card_w + gap),
            4.42,
            card_w,
            1.28,
            value=val,
            label=lbl,
            sub=sub,
            accent=acc,
            bg=C_DARK_CARD,
            border=C_DARK_BORDER,
            dark=True,
        )

    # Bottom metadata strip
    add_textbox(
        slide,
        0.80,
        6.45,
        11.73,
        0.35,
        text="Course Workshop Presentation (15 min)  ·  UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012)",
        font_size=10.5,
        color=C_TEXT_MUTED_DARK,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    set_notes(
        slide,
        "Slide 1 (~0:45): Bună ziua! Astăzi prezint proiectul 2.6: Gas Sensor Array Drift. "
        "Obiectivul este clasificarea multi-clasă a 6 gaze pe baza unei rețele de 16 senzori chimici MOX "
        "care suferă de drift temporal pe parcursul a 36 de luni. Am implementat un baseline MLP și toate "
        "cele 3 extensii: comparație cu 1D-CNN, validare cronologică pe batch-uri și reducere de dimensionalitate cu UMAP.",
    )


def build_slide_02_problem(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        2,
        "Problem & Motivation",
        "Electronic-Nose Sensing Fails Without Drift-Aware Evaluation",
    )

    cards = [
        (
            "1. Electronic Nose Array",
            C_BLUE,
            C_BLUE_TINT,
            [
                ("16 MOX sensors: ", "Cross-sensitive metal-oxide elements measure conductance changes upon gas exposure."),
                ("Compact & low-cost: ", "Enables real-time industrial safety, food quality, and environmental monitoring."),
                ("128 descriptors: ", "Steady-state response, normalized DR, and 6 exponential moving averages per sensor."),
            ],
        ),
        (
            "2. Physical Sensor Drift",
            C_AMBER,
            C_AMBER_TINT,
            [
                ("Gradual degradation: ", "Sensor poisoning, electrode aging, thermal cycling, and surface contamination."),
                ("36-month horizon: ", "Baseline conductance decays nonlinearly across 10 chronological collection batches."),
                ("No easy hardware fix: ", "Frequent laboratory recalibration is expensive or infeasible after deployment."),
            ],
        ),
        (
            "3. Covariate Shift Challenge",
            C_CORAL,
            C_CORAL_TINT,
            [
                ("Distribution shift: ", "P_train(X) differs sharply from P_test(X) on future unseen batches."),
                ("Random CV illusion: ", "Standard K-Fold mixes Months 1–36, leaking drift patterns into training."),
                ("True requirement: ", "Models must generalize forward in time without seeing future batch distributions."),
            ],
        ),
    ]

    card_w = 3.84
    gap = 0.30
    for idx, (ctitle, accent, tint, bullets) in enumerate(cards):
        cx = 0.60 + idx * (card_w + gap)
        cy = 1.20
        ch = 3.75
        add_rect(
            slide, cx, cy, card_w, ch, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True
        )
        add_rect(slide, cx, cy, card_w, 0.08, fill_color=accent)
        add_rect(slide, cx + 0.18, cy + 0.22, card_w - 0.36, 0.38, fill_color=tint, rounded=True)
        add_textbox(
            slide,
            cx + 0.26,
            cy + 0.22,
            card_w - 0.52,
            0.38,
            text=ctitle,
            font_size=13.0,
            bold=True,
            color=accent,
            valign=MSO_ANCHOR.MIDDLE,
            margin_lr=0.0,
            margin_tb=0.0,
        )
        add_bullets(
            slide,
            cx + 0.18,
            cy + 0.74,
            card_w - 0.36,
            2.85,
            bullets,
            font_size=11.0,
            space_after=10.0,
        )

    # Bottom dark callout banner
    add_rect(slide, 0.60, 5.18, 12.133, 1.72, fill_color=C_DARK_BG, rounded=True)
    add_rect(slide, 0.60, 5.18, 0.10, 1.72, fill_color=C_TEAL_BRIGHT)
    add_textbox(
        slide,
        0.92,
        5.30,
        11.50,
        0.32,
        text="THREE CORE RESEARCH QUESTIONS",
        font_size=11.0,
        bold=True,
        color=C_TEAL_BRIGHT,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_bullets(
        slide,
        0.92,
        5.66,
        11.50,
        1.15,
        [
            ("RQ1 (Temporal CV): ", "How severely does Random 5-Fold CV overestimate accuracy compared to chronological expanding-window CV?"),
            ("RQ2 (MLP vs. 1D-CNN): ", "Does reshaping 128 features into an 8×16 sensor sequence improve resilience to temporal drift?"),
            ("RQ3 (UMAP Compression): ", "Does projecting data onto a 16-D UMAP manifold fitted strictly on past batches help or hurt generalization?"),
        ],
        font_size=10.5,
        lead_color=C_TEXT_LIGHT,
        body_color=C_TEXT_MUTED_DARK,
        space_after=4.0,
    )

    set_notes(
        slide,
        "Slide 2 (~1:00): Nasurile electronice folosesc senzori MOX ieftini, dar suferă de drift fizic în timp "
        "(îmbătrânire, contaminare, variații termice). În ML asta înseamnă covariate shift: distribuția datelor de test "
        "diferă de cea de train. Dacă facem K-Fold aleatoriu, amestecăm viitorul cu trecutul. De aici cele 3 întrebări de cercetare.",
    )


def build_slide_03_dataset(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        3,
        "Dataset Architecture",
        "13,910 Samples Across 10 Imbalanced Chronological Batches",
    )

    kpis = [
        ("13,910", "Clean Measurements", "0 NaN · 0 Inf · 0 Duplicates", C_TEAL),
        ("16 × 8", "128 Sensor Features", "Steady-state + 6 EMA descriptors", C_BLUE),
        ("[-23.7K, +670.7K]", "Raw Scale Span", "Mandates Z-score standardization", C_AMBER),
        ("0 Toluene", "Missing in Batches 3–5", "Batch sizes: 161 (B4) to 3,613 (B7)", C_CORAL),
    ]
    kw = 2.85
    kgap = 0.244
    for idx, (val, lbl, sub, acc) in enumerate(kpis):
        add_kpi_card(slide, 0.60 + idx * (kw + kgap), 1.16, kw, 1.15, val, lbl, sub, accent=acc)

    add_fitted_picture(
        slide,
        FIGURES_DIR / "01_batch_class_distribution.png",
        box_x=0.60,
        box_y=2.46,
        box_w=12.133,
        box_h=4.52,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 3 (~1:15): Dataset-ul UCI are 13.910 măsurători, 128 de caracteristici și 6 clase de gaze "
        "(Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene), organizate cronologic în 10 batch-uri. "
        "Observați două provocări critice în heatmap: batch-urile sunt foarte dezechilibrate (de la 161 în Batch 4 la 3.613 în Batch 7), "
        "iar Toluene lipsește complet din Batch-urile 3, 4 și 5.",
    )


def build_slide_04_sensor_drift(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        4,
        "Exploratory Drift Analysis",
        "Steady-State Sensor Responses Decay and Shift Over 36 Months",
    )

    left_cards = [
        (
            "Severe Baseline Decay (B1 → B8)",
            C_CORAL,
            "Acetone mean steady-state DR on Sensor 1 drops from ~396K in Batch 1 down toward near-zero in Batch 8, compressing inter-class margins.",
        ),
        (
            "Non-Monotonic Regime Shift (B10)",
            C_AMBER,
            "In Batch 10 (Month 36, n=3,600), sensor responses rebound sharply with high variance, creating an out-of-distribution test regime.",
        ),
        (
            "Strict Train-Only Z-Score Scaling",
            C_TEAL,
            "Raw features span 5+ orders of magnitude. StandardScaler must be fitted strictly on training batches 1..t to prevent future leakage.",
        ),
    ]

    for idx, (ctitle, accent, desc) in enumerate(left_cards):
        cy = 1.18 + idx * 1.98
        ch = 1.82
        add_rect(slide, 0.60, cy, 4.35, ch, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True)
        add_rect(slide, 0.60, cy, 0.08, ch, fill_color=accent)
        add_textbox(
            slide, 0.82, cy + 0.16, 3.98, 0.34, text=ctitle, font_size=13.0, bold=True, color=accent, margin_lr=0.0, margin_tb=0.0
        )
        add_textbox(
            slide, 0.82, cy + 0.56, 3.98, 1.12, text=desc, font_size=11.0, color=C_TEXT_SEC, margin_lr=0.0, margin_tb=0.0
        )

    add_fitted_picture(
        slide,
        FIGURES_DIR / "02_sensor_drift_over_batches.png",
        box_x=5.15,
        box_y=1.18,
        box_w=7.58,
        box_h=5.80,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 4 (~1:15): Aici vedem drift-ul fizic pe 4 senzori reprezentativi (f1, f9, f17, f25). "
        "Valorile medii scad drastic din Batch 1 până în Batch 8, apoi cresc din nou în Batch 10. "
        "De aceea standardizarea Z-score este obligatorie, dar trebuie calculată strict pe fold-ul de antrenare.",
    )


def build_slide_05_pca(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        5,
        "Feature Space Geometry",
        "2D PCA (68.6% Variance) Reveals Systematic Batch-to-Batch Drift",
    )

    pills = [
        ("68.6% Variance Captured", "PC1 (53.5%) + PC2 (15.0%) on standardized 128-D features", C_BLUE, C_BLUE_TINT),
        ("Distinct Gas Manifolds", "Left: Ethylene, Acetone, and Ethanol form coherent branches", C_TEAL, C_TEAL_TINT),
        ("Chronological Shift", "Right: Batch 1 (dark) vs. Batch 10 (yellow) occupy shifted regions", C_AMBER, C_AMBER_TINT),
    ]
    pw = 3.88
    pgap = 0.246
    for idx, (ptitle, psub, acc, tint) in enumerate(pills):
        px = 0.60 + idx * (pw + pgap)
        add_rect(slide, px, 1.16, pw, 1.05, fill_color=tint, border_color=acc, border_pt=1.0, rounded=True)
        add_textbox(
            slide, px + 0.16, 1.24, pw - 0.32, 0.30, text=ptitle, font_size=12.5, bold=True, color=acc, margin_lr=0.0, margin_tb=0.0
        )
        add_textbox(
            slide, px + 0.16, 1.56, pw - 0.32, 0.56, text=psub, font_size=10.5, color=C_TEXT_DARK, margin_lr=0.0, margin_tb=0.0
        )

    add_fitted_picture(
        slide,
        FIGURES_DIR / "03_pca_drift_visualization.png",
        box_x=0.60,
        box_y=2.36,
        box_w=12.133,
        box_h=4.62,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 5 (~1:00): Proiecția PCA 2D explică 68.6% din varianță și confirmă vizual problema: "
        "în stânga vedem separarea gazelor, dar în dreapta, colorat după batch-ul cronologic, "
        "observăm clar că Batch 1 (mov) și Batch 10 (galben) ocupă zone deplasate în spațiul caracteristicilor.",
    )


def build_slide_06_architectures(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        6,
        "Deep Learning Methodology",
        "Model Architectures: GasMLP Baseline vs. Gas1DCNN (Extension 1)",
    )

    # Left Card: GasMLP
    lx, ly, lw, lh = 0.60, 1.18, 5.90, 4.35
    add_rect(slide, lx, ly, lw, lh, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True)
    add_rect(slide, lx, ly, lw, 0.08, fill_color=C_BLUE)
    add_textbox(
        slide, lx + 0.22, ly + 0.16, lw - 0.44, 0.32, text="GasMLP Baseline (Flat 128-D Vector)", font_size=14.0, bold=True, color=C_BLUE, margin_lr=0.0, margin_tb=0.0
    )

    mlp_blocks = [
        ("Input Layer", "128 standardized features (or 16-D UMAP)", C_BLUE_TINT, C_BLUE),
        ("Hidden Block 1", "Linear(D -> 256) + BatchNorm1d + ReLU + Dropout(0.3)", C_LIGHT_BG, C_BORDER),
        ("Hidden Block 2", "Linear(256 -> 128) + BatchNorm1d + ReLU + Dropout(0.3)", C_LIGHT_BG, C_BORDER),
        ("Hidden Block 3", "Linear(128 -> 64) + BatchNorm1d + ReLU + Dropout(0.3)", C_LIGHT_BG, C_BORDER),
        ("Classification Head", "Linear(64 -> 6 gas logits)", C_BLUE_TINT, C_BLUE),
    ]
    for b_idx, (bt, bs, bfill, bborder) in enumerate(mlp_blocks):
        by = ly + 0.58 + b_idx * 0.72
        add_flow_box(slide, lx + 0.35, by, lw - 0.70, 0.58, bt, bs, bfill, bborder)
        if b_idx < len(mlp_blocks) - 1:
            add_textbox(
                slide, lx, by + 0.56, lw, 0.18, text="▼", font_size=8.5, bold=True, color=C_BLUE, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0
            )

    # Right Card: Gas1DCNN
    rx, ry, rw, rh = 6.833, 1.18, 5.90, 4.35
    add_rect(slide, rx, ry, rw, rh, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True)
    add_rect(slide, rx, ry, rw, 0.08, fill_color=C_TEAL)
    add_textbox(
        slide, rx + 0.22, ry + 0.16, rw - 0.44, 0.32, text="Gas1DCNN — Extension 1 (Sensor Sequence)", font_size=14.0, bold=True, color=C_TEAL, margin_lr=0.0, margin_tb=0.0
    )

    cnn_blocks = [
        ("Tensor Reshape", "(B, 128) -> (B, 16, 8) -> permute (B, 8 channels, 16 sensors)", C_TEAL_TINT, C_TEAL),
        ("Conv Block 1 & 2", "Conv1d(8->32, k=3, p=1) + BN + ReLU -> Conv1d(32->64) + BN + ReLU", C_LIGHT_BG, C_BORDER),
        ("Conv Block 3", "Conv1d(64->128, k=3, p=1) + BatchNorm1d + ReLU", C_LIGHT_BG, C_BORDER),
        ("Dual Global Pooling", "Concat[AdaptiveAvgPool1d(1), AdaptiveMaxPool1d(1)] -> 256-D", C_LIGHT_BG, C_BORDER),
        ("Classifier Head", "Dropout(0.3) -> Linear(256->64) -> ReLU -> Dropout(0.2) -> 6 logits", C_TEAL_TINT, C_TEAL),
    ]
    for b_idx, (bt, bs, bfill, bborder) in enumerate(cnn_blocks):
        by = ry + 0.58 + b_idx * 0.72
        add_flow_box(slide, rx + 0.35, by, rw - 0.70, 0.58, bt, bs, bfill, bborder)
        if b_idx < len(cnn_blocks) - 1:
            add_textbox(
                slide, rx, by + 0.56, rw, 0.18, text="▼", font_size=8.5, bold=True, color=C_TEAL, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0
            )

    # Bottom shared training config banner
    add_rect(slide, 0.60, 5.70, 12.133, 1.28, fill_color=C_DARK_BG, rounded=True)
    add_rect(slide, 0.60, 5.70, 0.10, 1.28, fill_color=C_TEAL_BRIGHT)
    add_textbox(
        slide,
        0.92,
        5.82,
        11.50,
        0.28,
        text="UNIFIED OPTIMIZATION & REGULARIZATION PROTOCOL",
        font_size=10.5,
        bold=True,
        color=C_TEAL_BRIGHT,
        margin_lr=0.0,
        margin_tb=0.0,
    )
    add_textbox(
        slide,
        0.92,
        6.16,
        11.50,
        0.64,
        text=(
            "Loss: Multi-class CrossEntropyLoss   |   Optimizer: AdamW (lr = 1e-3, weight_decay = 1e-4)   |   "
            "Schedule: CosineAnnealingLR (T_max = 60)\n"
            "Early Stopping: patience = 12 on internal 15% stratified validation split (best validation checkpoint restored)   |   Batch size = 128"
        ),
        font_size=11.0,
        color=C_TEXT_LIGHT,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    set_notes(
        slide,
        "Slide 6 (~1:30): Comparăm două arhitecturi în PyTorch. Baseline-ul GasMLP primește vectorul de 128 de valori "
        "prin 3 blocuri Dense + BatchNorm + ReLU + Dropout(0.3). Pentru extensia 1D-CNN, am reorganizat cele 128 de caracteristici "
        "în 8 canale peste secvența celor 16 senzori, aplicând 3 blocuri Conv1d și Dual Global Pooling (Avg + Max). "
        "Ambele modele sunt antrenate identic cu AdamW, CosineAnnealingLR și Early Stopping.",
    )


def build_slide_07_validation(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        7,
        "Extension 2 · Evaluation Protocol",
        "Replacing Leaky Random CV With Chronological Expanding-Window CV",
    )

    # Top Card: Protocol A (Naive)
    ax, ay, aw, ah = 0.60, 1.20, 12.133, 2.55
    add_rect(slide, ax, ay, aw, ah, fill_color=C_WHITE, border_color=C_CORAL, border_pt=1.2, rounded=True)
    add_rect(slide, ax, ay, 0.10, ah, fill_color=C_CORAL)
    add_textbox(
        slide, ax + 0.28, ay + 0.16, 8.20, 0.34, text="Protocol A — Naive Random Stratified 5-Fold CV (Leaky Baseline)", font_size=14.0, bold=True, color=C_CORAL, margin_lr=0.0, margin_tb=0.0
    )
    add_rect(slide, ax + 9.45, ay + 0.16, 2.40, 0.34, fill_color=C_CORAL_TINT, border_color=C_CORAL, border_pt=1.0, rounded=True)
    add_textbox(
        slide, ax + 9.45, ay + 0.16, 2.40, 0.34, text="OPTIMISTIC BIAS", font_size=9.5, bold=True, color=C_CORAL, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
    )

    # Fold visual blocks for Protocol A
    fold_w = 2.15
    fold_gap = 0.20
    for f_i in range(5):
        fx = ax + 0.28 + f_i * (fold_w + fold_gap)
        is_test = f_i == 4
        fc = C_CORAL_TINT if is_test else C_LIGHT_BG
        bc = C_CORAL if is_test else C_BORDER
        lbl = f"Fold {f_i+1}: Test (20%)\nMixes B1–B10" if is_test else f"Fold {f_i+1}: Train (80%)\nMixes B1–B10"
        add_rect(slide, fx, ay + 0.68, fold_w, 0.85, fill_color=fc, border_color=bc, border_pt=1.0, rounded=True)
        add_textbox(
            slide, fx + 0.08, ay + 0.72, fold_w - 0.16, 0.76, text=lbl, font_size=10.5, bold=is_test, color=C_CORAL if is_test else C_TEXT_DARK, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
        )

    add_textbox(
        slide,
        ax + 0.28,
        ay + 1.72,
        11.55,
        0.68,
        text=(
            "Why it fails: Randomly shuffling all 13,910 samples exposes the model to drifted measurements from Months 24–36 during training. "
            "Models memorize batch-specific signatures instead of generalizing forward in time."
        ),
        font_size=11.0,
        color=C_TEXT_SEC,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    # Bottom Card: Protocol B (Temporal)
    bx, by, bw, bh = 0.60, 3.95, 12.133, 2.95
    add_rect(slide, bx, by, bw, bh, fill_color=C_WHITE, border_color=C_EMERALD, border_pt=1.2, rounded=True)
    add_rect(slide, bx, by, 0.10, bh, fill_color=C_EMERALD)
    add_textbox(
        slide, bx + 0.28, by + 0.16, 8.60, 0.34, text="Protocol B — Chronological Expanding-Window CV (Extension 2)", font_size=14.0, bold=True, color=C_EMERALD, margin_lr=0.0, margin_tb=0.0
    )
    add_rect(slide, bx + 9.45, by + 0.16, 2.40, 0.34, fill_color=C_EMERALD_TINT, border_color=C_EMERALD, border_pt=1.0, rounded=True)
    add_textbox(
        slide, bx + 9.45, by + 0.16, 2.40, 0.34, text="ZERO TEMPORAL LEAKAGE", font_size=9.5, bold=True, color=C_EMERALD, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
    )

    # Expanding window timeline rows
    examples = [
        ("Step t=1", 1, 2),
        ("Step t=5", 5, 6),
        ("Step t=9", 9, 10),
    ]
    cell_w = 0.96
    cell_gap = 0.08
    for r_idx, (step_lbl, tr_end, te_b) in enumerate(examples):
        ry = by + 0.68 + r_idx * 0.44
        add_textbox(
            slide, bx + 0.28, ry, 1.05, 0.34, text=step_lbl, font_size=10.5, bold=True, color=C_TEXT_DARK, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
        )
        for b in range(1, 11):
            cx = bx + 1.40 + (b - 1) * (cell_w + cell_gap)
            if b <= tr_end:
                cfill, cbor, ctxt, bld = C_TEAL_TINT, C_TEAL, C_TEAL, True
                label = f"B{b} Train"
            elif b == te_b:
                cfill, cbor, ctxt, bld = C_AMBER_TINT, C_AMBER, C_AMBER, True
                label = f"B{b} Test"
            else:
                cfill, cbor, ctxt, bld = C_LIGHT_BG, C_DIVIDER, C_TEXT_FOOTER, False
                label = f"B{b}"
            add_rect(slide, cx, ry, cell_w, 0.34, fill_color=cfill, border_color=cbor, border_pt=0.9, rounded=True)
            add_textbox(
                slide, cx, ry, cell_w, 0.34, text=label, font_size=8.8, bold=bld, color=ctxt, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
            )

    add_textbox(
        slide,
        bx + 0.28,
        by + 2.08,
        11.55,
        0.74,
        text=(
            "Leakage-Free Guarantee: At every step t = 1..9, StandardScaler and UMAP are fitted strictly on Batches 1..t and evaluated on unseen Batch t+1. "
            "Macro F1 is evaluated over classes present in each test batch, alongside pooled metrics over all 13,465 test predictions (Batches 2–10)."
        ),
        font_size=10.8,
        color=C_TEXT_SEC,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    set_notes(
        slide,
        "Slide 7 (~1:15): Diferența dintre cele două protocoale de validare: Protocolul A (Random 5-Fold CV) amestecă "
        "eșantioane din toate cele 36 de luni și produce o estimare nerealist de optimistă. Protocolul B (Chronological Expanding-Window CV) "
        "antrenează exclusiv pe batch-urile trecute 1..t și testează pe batch-ul viitor t+1, cu StandardScaler și UMAP ajustate strict pe train.",
    )


def build_slide_08_umap(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        8,
        "Extension 3 · Manifold Compression",
        "Non-Linear UMAP Projection Captures Local Clusters but Struggles Out-of-Sample",
    )

    cards = [
        ("Fuzzy Simplicial Graph", "n_neighbors = 15, min_dist = 0.1, Euclidean metric on standardized 128-D space", C_TEAL, C_TEAL_TINT),
        ("2D Exploratory Projection", "Visualizes local sub-manifolds by gas class (left) and drift trajectories across batches (right)", C_BLUE, C_BLUE_TINT),
        ("16-D UMAP+MLP Pipeline", "16-D UMAP fitted strictly on Batches 1..t; future Batch t+1 projected via .transform() into GasMLP", C_AMBER, C_AMBER_TINT),
    ]
    cw = 3.88
    cgap = 0.246
    for idx, (ct, cs, acc, tint) in enumerate(cards):
        cx = 0.60 + idx * (cw + cgap)
        add_rect(slide, cx, 1.16, cw, 1.05, fill_color=tint, border_color=acc, border_pt=1.0, rounded=True)
        add_textbox(
            slide, cx + 0.16, 1.24, cw - 0.32, 0.30, text=ct, font_size=12.0, bold=True, color=acc, margin_lr=0.0, margin_tb=0.0
        )
        add_textbox(
            slide, cx + 0.16, 1.56, cw - 0.32, 0.56, text=cs, font_size=10.0, color=C_TEXT_DARK, margin_lr=0.0, margin_tb=0.0
        )

    add_fitted_picture(
        slide,
        FIGURES_DIR / "04_umap_2d_visualization.png",
        box_x=0.60,
        box_y=2.36,
        box_w=12.133,
        box_h=4.62,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 8 (~1:00): Extensia 3 aplică UMAP. În scop exploratoriu (2D), UMAP separă foarte clar clusterele locale "
        "ale gazelor, dar observăm și fragmentarea fiecărui gaz în sub-clustere corespunzătoare batch-urilor (dreapta). "
        "Pentru clasificare (UMAP+MLP), reducem la 16 dimensiuni ajustate strict pe batch-urile de antrenare 1..t.",
    )


def build_slide_09_random_vs_temporal(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        9,
        "Core Finding 1 · Impact of Drift",
        "Chronological Evaluation Exposes a ~17% Accuracy Drop Masked by Random CV",
    )

    kpis = [
        ("99.31% → 82.57%", "MLP Accuracy Penalty (-16.74 pp)", "Macro F1 drops from 99.28% to 78.66%", C_BLUE),
        ("99.35% → 81.92%", "1D-CNN Accuracy Penalty (-17.43 pp)", "Macro F1 drops from 99.32% to 77.72%", C_TEAL),
        ("71.83% Mean Acc", "UMAP+MLP (16-D) Under Drift", "Pooled Acc: 67.13% · Pooled F1: 65.96%", C_CORAL),
    ]
    kw = 3.88
    kgap = 0.246
    for idx, (val, lbl, sub, acc) in enumerate(kpis):
        add_kpi_card(slide, 0.60 + idx * (kw + kgap), 1.16, kw, 1.22, val, lbl, sub, accent=acc)

    add_fitted_picture(
        slide,
        FIGURES_DIR / "05_random_vs_temporal_cv.png",
        box_x=0.60,
        box_y=2.52,
        box_w=12.133,
        box_h=4.46,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 9 (~1:30): Acesta este rezultatul central al proiectului. Pe 5-Fold CV aleatoriu, atât MLP cât și 1D-CNN "
        "obțin peste 99.3% acuratețe. Dar sub validare temporală realistă, performanța scade cu ~17 puncte procentuale: "
        "82.57% pentru MLP și 81.92% pentru 1D-CNN, în timp ce UMAP+MLP ajunge la 71.83%. "
        "Asta demonstrează cât de periculoasă este validarea ne-temporală pe date cu drift.",
    )


def build_slide_10_benchmark_table(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        10,
        "Experimental Benchmark",
        "Full-Feature GasMLP Outperforms 1D-CNN and UMAP+MLP Under Temporal Drift",
    )

    rand_df = pd.read_csv(RESULTS_DIR / "random_cv_summary.csv").set_index("model")
    temp_df = pd.read_csv(RESULTS_DIR / "temporal_cv_summary.csv").set_index("model")

    headers = [
        "Protocol",
        "Model",
        "Input Dim",
        "Mean Acc (± std)",
        "Mean Macro F1 (± std)",
        "Pooled Acc",
        "Pooled Macro F1",
        "Pooled W-F1",
    ]
    rows_data = [
        [
            "Random 5-Fold CV",
            "MLP",
            "128-D",
            f"{rand_df.loc['MLP', 'accuracy_mean']*100:.2f}% ± {rand_df.loc['MLP', 'accuracy_std']*100:.2f}%",
            f"{rand_df.loc['MLP', 'macro_f1_mean']*100:.2f}% ± {rand_df.loc['MLP', 'macro_f1_std']*100:.2f}%",
            "—",
            "—",
            "—",
        ],
        [
            "Random 5-Fold CV",
            "1D-CNN",
            "8 × 16",
            f"{rand_df.loc['1D-CNN', 'accuracy_mean']*100:.2f}% ± {rand_df.loc['1D-CNN', 'accuracy_std']*100:.2f}%",
            f"{rand_df.loc['1D-CNN', 'macro_f1_mean']*100:.2f}% ± {rand_df.loc['1D-CNN', 'macro_f1_std']*100:.2f}%",
            "—",
            "—",
            "—",
        ],
        [
            "Temporal Expanding CV",
            "MLP (Best Temporal)",
            "128-D",
            f"{temp_df.loc['MLP', 'mean_batch_accuracy']*100:.2f}% ± {temp_df.loc['MLP', 'std_batch_accuracy']*100:.2f}%",
            f"{temp_df.loc['MLP', 'mean_batch_macro_f1']*100:.2f}% ± {temp_df.loc['MLP', 'std_batch_macro_f1']*100:.2f}%",
            f"{temp_df.loc['MLP', 'pooled_accuracy']*100:.2f}%",
            f"{temp_df.loc['MLP', 'pooled_macro_f1']*100:.2f}%",
            f"{temp_df.loc['MLP', 'pooled_weighted_f1']*100:.2f}%",
        ],
        [
            "Temporal Expanding CV",
            "1D-CNN",
            "8 × 16",
            f"{temp_df.loc['1D-CNN', 'mean_batch_accuracy']*100:.2f}% ± {temp_df.loc['1D-CNN', 'std_batch_accuracy']*100:.2f}%",
            f"{temp_df.loc['1D-CNN', 'mean_batch_macro_f1']*100:.2f}% ± {temp_df.loc['1D-CNN', 'std_batch_macro_f1']*100:.2f}%",
            f"{temp_df.loc['1D-CNN', 'pooled_accuracy']*100:.2f}%",
            f"{temp_df.loc['1D-CNN', 'pooled_macro_f1']*100:.2f}%",
            f"{temp_df.loc['1D-CNN', 'pooled_weighted_f1']*100:.2f}%",
        ],
        [
            "Temporal Expanding CV",
            "UMAP + MLP",
            "16-D",
            f"{temp_df.loc['UMAP+MLP', 'mean_batch_accuracy']*100:.2f}% ± {temp_df.loc['UMAP+MLP', 'std_batch_accuracy']*100:.2f}%",
            f"{temp_df.loc['UMAP+MLP', 'mean_batch_macro_f1']*100:.2f}% ± {temp_df.loc['UMAP+MLP', 'std_batch_macro_f1']*100:.2f}%",
            f"{temp_df.loc['UMAP+MLP', 'pooled_accuracy']*100:.2f}%",
            f"{temp_df.loc['UMAP+MLP', 'pooled_macro_f1']*100:.2f}%",
            f"{temp_df.loc['UMAP+MLP', 'pooled_weighted_f1']*100:.2f}%",
        ],
    ]

    tbl_shape = slide.shapes.add_table(
        len(rows_data) + 1, len(headers), Inches(0.60), Inches(1.22), Inches(12.133), Inches(3.55)
    )
    table = tbl_shape.table
    col_widths = [2.05, 1.80, 0.92, 1.84, 1.88, 1.12, 1.36, 1.163]
    for ci, cw in enumerate(col_widths):
        table.columns[ci].width = Inches(cw)

    # Header row
    for ci, htxt in enumerate(headers):
        cell = table.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_DARK_BG
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if ci >= 2 else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = htxt
        r.font.name = FONT_NAME
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = C_TEXT_LIGHT

    # Data rows
    for ri, rdata in enumerate(rows_data, start=1):
        is_best_temporal = ri == 3
        row_bg = C_EMERALD_TINT if is_best_temporal else (C_WHITE if ri % 2 == 1 else C_LIGHT_BG)
        for ci, val in enumerate(rdata):
            cell = table.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci >= 2 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = FONT_NAME
            r.font.size = Pt(10.5)
            r.font.bold = is_best_temporal and ci in (1, 3, 4, 5, 6)
            r.font.color.rgb = C_EMERALD if (is_best_temporal and ci in (1, 5, 6)) else C_TEXT_DARK

    # Bottom 2 synthesis cards
    synth = [
        (
            "Why Global Dense MLP Edges Out Local 1D-CNN",
            C_BLUE,
            [
                ("Heterogeneous sensor coatings: ", "The 16 MOX sensors have distinct chemical sensitivities rather than strict 1D translation invariance."),
                ("Global cross-sensor weights: ", "Full 128-D connections in GasMLP adapt more stably to late-stage Batch 10 drift (77.25% vs. 67.92%)."),
            ],
        ),
        (
            "Why Out-of-Sample UMAP Degrades Under Shift",
            C_CORAL,
            [
                ("Neighborhood fragility: ", "UMAP .transform() interpolates coordinates from k-NN graphs built on training Batches 1..t."),
                ("Support mismatch: ", "When future drifted samples fall outside training support, manifold projections distort class boundaries (67.13% pooled)."),
            ],
        ),
    ]
    sw = 5.92
    sgap = 0.293
    for idx, (stitle, sacc, sbullets) in enumerate(synth):
        sx = 0.60 + idx * (sw + sgap)
        sy = 4.98
        sh = 1.98
        add_rect(slide, sx, sy, sw, sh, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True)
        add_rect(slide, sx, sy, 0.08, sh, fill_color=sacc)
        add_textbox(
            slide, sx + 0.24, sy + 0.14, sw - 0.40, 0.32, text=stitle, font_size=13.0, bold=True, color=sacc, margin_lr=0.0, margin_tb=0.0
        )
        add_bullets(slide, sx + 0.24, sy + 0.54, sw - 0.40, 1.32, sbullets, font_size=10.8, space_after=8.0)

    set_notes(
        slide,
        "Slide 10 (~1:15): Tabelul complet al rezultatelor. Sub validare temporală, MLP-ul pe toate cele 128 de caracteristici "
        "câștigă atât ca medie pe batch-uri (82.57% acuratețe, 78.66% Macro F1), cât și per total agregat (80.31% acuratețe pe cele 13.465 de predicții). "
        "Două explicații cheie: senzorii MOX sunt eterogeni (conexiunile globale ale MLP-ului se adaptează mai bine decât nucleele locale 1D), "
        "iar proiecția UMAP out-of-sample devine fragilă când eșantioanele viitoare ies din suportul de antrenare.",
    )


def build_slide_11_per_batch(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        11,
        "Core Finding 2 · Per-Batch Dynamics",
        "1D-CNN Excels on Early Transitions; MLP Leads on Severe Batch 10 Drift",
    )

    kpis = [
        ("93.88% & 81.06%", "1D-CNN Wins Batches 3 & 9", "Beats MLP on B3 (79.32%) & B9 (76.81%)", C_TEAL),
        ("76.65% Dip at B6", "Toluene Reappearance Bottleneck", "Absent in B3–B5 -> stale B1–B2 signatures", C_AMBER),
        ("77.25% vs. 67.92%", "MLP Resilience on Batch 10", "Largest test batch (n=3,600) at Month 36", C_BLUE),
    ]
    kw = 3.88
    kgap = 0.246
    for idx, (val, lbl, sub, acc) in enumerate(kpis):
        add_kpi_card(slide, 0.60 + idx * (kw + kgap), 1.16, kw, 1.22, val, lbl, sub, accent=acc)

    add_fitted_picture(
        slide,
        FIGURES_DIR / "06_temporal_accuracy_per_batch.png",
        box_x=0.60,
        box_y=2.52,
        box_w=12.133,
        box_h=4.46,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 11 (~1:30): Urmărind evoluția pas cu pas de la Batch 2 la Batch 10, vedem comportamente complementare: "
        "1D-CNN câștigă spectaculos în Batch 3 (93.88%) și Batch 9 (81.06%). Toate modelele scad la Batch 6 pentru că "
        "Toluene reapare după ce a lipsit complet în Batch-urile 3–5. În final, pe Batch 10 (3.600 de probe la luna 36), "
        "MLP își păstrează avantajul cu 77.25% față de 67.92% la 1D-CNN și 51.67% la UMAP+MLP.",
    )


def build_slide_12_confusion(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        12,
        "Error Analysis",
        "Confusion Matrices Pinpoint Toluene–Acetaldehyde Drift Overlap",
    )

    kpis = [
        ("90%–95% Recall", "Drift-Resilient Gases (MLP)", "Ethylene 95.0% · Acetone 92.8% · Ammonia 90.2%", C_EMERALD),
        ("37.0% Misclassified", "Dominant Error: Toluene -> Acetaldehyde", "Toluene recall drops to 55.2% (MLP) / 54.3% (CNN)", C_CORAL),
        ("21.7% Overlap", "Acetaldehyde -> Ethylene Confusion", "Driven by overlapping aged-sensor trajectories", C_AMBER),
    ]
    kw = 3.88
    kgap = 0.246
    for idx, (val, lbl, sub, acc) in enumerate(kpis):
        add_kpi_card(slide, 0.60 + idx * (kw + kgap), 1.16, kw, 1.15, val, lbl, sub, accent=acc)

    add_fitted_picture(
        slide,
        FIGURES_DIR / "07_confusion_matrices_temporal.png",
        box_x=0.60,
        box_y=2.45,
        box_w=12.133,
        box_h=4.53,
        pad=0.14,
    )

    set_notes(
        slide,
        "Slide 12 (~1:15): Analizând matricile de confuzie agregate pe Batch-urile 2–10: Ethylene (95%), Acetone (92.8%) "
        "și Ammonia (90.2%) rămân foarte stabile sub drift. Problema majoră este la Toluene și Acetaldehyde: "
        "37% din probele de Toluene sunt clasificate drept Acetaldehyde de către MLP — exact din cauza absenței Toluene în "
        "Batch-urile 3–5 și a suprapunerii traiectoriilor odată cu îmbătrânirea senzorilor.",
    )


def build_slide_13_software(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header_footer(
        slide,
        13,
        "Software Design & Reproducibility",
        "Modular PyTorch Codebase With Exact Seed and Hyperparameter Tracking",
    )

    # Top Pipeline Diagram: 4 connected module cards
    modules = [
        ("1. src/data_loader.py", "Downloads UCI archive, parses 10 LIBSVM batches, caches Parquet", C_BLUE, C_BLUE_TINT),
        ("2. src/models.py", "Defines GasMLP and Gas1DCNN PyTorch nn.Module architectures", C_TEAL, C_TEAL_TINT),
        ("3. src/train_eval.py", "AdamW, CosineLR, early stopping, Random CV & Expanding Temporal CV", C_EMERALD, C_EMERALD_TINT),
        ("4. notebooks/ 01 & 02", "Generates all 7 figures (300 DPI) + CSV/JSON benchmark tables", C_AMBER, C_AMBER_TINT),
    ]
    mw = 2.78
    mgap = 0.337
    for idx, (mt, ms, macc, mtint) in enumerate(modules):
        mx = 0.60 + idx * (mw + mgap)
        add_rect(slide, mx, 1.22, mw, 2.05, fill_color=C_WHITE, border_color=macc, border_pt=1.2, rounded=True)
        add_rect(slide, mx, 1.22, mw, 0.08, fill_color=macc)
        add_rect(slide, mx + 0.16, 1.42, mw - 0.32, 0.36, fill_color=mtint, rounded=True)
        add_textbox(
            slide, mx + 0.20, 1.42, mw - 0.40, 0.36, text=mt, font_size=11.0, bold=True, color=macc, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
        )
        add_textbox(
            slide, mx + 0.18, 1.90, mw - 0.36, 1.22, text=ms, font_size=10.5, color=C_TEXT_SEC, align=PP_ALIGN.CENTER, margin_lr=0.0, margin_tb=0.0
        )
        if idx < 3:
            add_textbox(
                slide, mx + mw, 2.02, mgap, 0.40, text="→", font_size=16.0, bold=True, color=C_TEAL, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_lr=0.0, margin_tb=0.0
            )

    # Bottom Left: Reproducibility & Environment Card
    lx, ly, lw, lh = 0.60, 3.55, 5.92, 3.40
    add_rect(slide, lx, ly, lw, lh, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True)
    add_rect(slide, lx, ly, lw, 0.08, fill_color=C_TEAL)
    add_textbox(
        slide, lx + 0.22, ly + 0.18, lw - 0.44, 0.32, text="Reproducibility & Execution Environment", font_size=13.5, bold=True, color=C_TEAL, margin_lr=0.0, margin_tb=0.0
    )
    add_bullets(
        slide,
        lx + 0.22,
        ly + 0.60,
        lw - 0.44,
        2.65,
        [
            ("Compute Stack: ", "PyTorch 2.6.0+cu124 on NVIDIA GeForce RTX 3050 Laptop GPU."),
            ("Deterministic Seeding: ", "Fixed seed = 42 across Python, NumPy, and PyTorch CUDA."),
            ("Tracked Config: ", "All settings exported to results/hyperparameters.json & CSVs."),
            ("One-Command Run: ", "jupyter nbconvert --execute on Notebooks 01 & 02 reproduces everything."),
        ],
        font_size=10.8,
        space_after=8.0,
    )

    # Bottom Right: Original Contributions vs Starting Point
    rx, ry, rw, rh = 6.813, 3.55, 5.92, 3.40
    add_rect(slide, rx, ry, rw, rh, fill_color=C_WHITE, border_color=C_BORDER, border_pt=1.0, rounded=True)
    add_rect(slide, rx, ry, rw, 0.08, fill_color=C_BLUE)
    add_textbox(
        slide, rx + 0.22, ry + 0.18, rw - 0.44, 0.32, text="Original Contributions vs. Starting Point", font_size=13.5, bold=True, color=C_BLUE, margin_lr=0.0, margin_tb=0.0
    )
    add_bullets(
        slide,
        rx + 0.22,
        ry + 0.60,
        rw - 0.44,
        2.65,
        [
            ("Starting Point [27]: ", "Dalageorgos (2025) baseline exploration on Gas Sensor Drift."),
            ("Custom PyTorch Models: ", "Built GasMLP + Gas1DCNN (8×16 sensor reshaping + dual pooling)."),
            ("Drift-Safe Harness: ", "Chronological expanding-window CV with strictly fold-fitted scalers."),
            ("UMAP & Full Reporting: ", "Leakage-free 16-D UMAP+MLP evaluation, 7 figures, 5,000-word-cap paper."),
        ],
        font_size=10.8,
        space_after=8.0,
    )

    set_notes(
        slide,
        "Slide 13 (~0:45): Din punct de vedere software, codul este complet modularizat în pachetul src/ și două notebook-uri "
        "reproductibile, rulate pe GPU RTX 3050 cu PyTorch 2.6.0+cu124 și seed fixat la 42. Pornind de la punctul de plecare [27], "
        "toate arhitecturile PyTorch, protocolul de validare cronologică și pipeline-ul UMAP au fost implementate de la zero.",
    )


def build_slide_14_conclusions(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, C_DARK_BG)

    add_rect(slide, 0.0, 0.0, SLIDE_W, 0.08, fill_color=C_TEAL_BRIGHT)
    add_textbox(
        slide, 0.65, 0.28, 10.50, 0.26, text="CONCLUSIONS & OUTLOOK", font_size=10.0, bold=True, color=C_TEAL_BRIGHT, margin_lr=0.0, margin_tb=0.0
    )
    add_textbox(
        slide, 0.65, 0.54, 11.20, 0.48, text="Key Takeaways & Future Drift-Adaptation Directions", font_size=22.0, bold=True, color=C_TEXT_LIGHT, margin_lr=0.0, margin_tb=0.0
    )
    add_textbox(
        slide, 11.95, 0.36, 0.78, 0.30, text=f"14 / {TOTAL_SLIDES}", font_size=9.5, bold=True, color=C_TEAL_BRIGHT, align=PP_ALIGN.RIGHT, margin_lr=0.0, margin_tb=0.0
    )

    # Left Column: 3 Key Conclusions
    add_rect(slide, 0.65, 1.26, 5.85, 4.65, fill_color=C_DARK_CARD, border_color=C_DARK_BORDER, border_pt=1.0, rounded=True)
    add_rect(slide, 0.65, 1.26, 5.85, 0.07, fill_color=C_TEAL_BRIGHT)
    add_textbox(
        slide, 0.90, 1.46, 5.35, 0.34, text="Key Conclusions", font_size=15.0, bold=True, color=C_TEAL_BRIGHT, margin_lr=0.0, margin_tb=0.0
    )
    add_bullets(
        slide,
        0.90,
        1.94,
        5.35,
        3.80,
        [
            ("1. Temporal CV Is Mandatory: ", "Random 5-Fold CV inflates accuracy to 99.31%–99.35%; expanding-window CV reveals true drift performance (~82%)."),
            ("2. GasMLP Leads Overall: ", "Full-feature MLP achieves 82.57% mean / 80.31% pooled accuracy and holds 77.25% on Month-36 Batch 10."),
            ("3. UMAP Struggles Out-of-Sample: ", "16-D UMAP+MLP drops to 67.13% pooled accuracy because k-NN embeddings break under covariate shift."),
        ],
        font_size=11.5,
        lead_color=C_TEXT_LIGHT,
        body_color=C_TEXT_MUTED_DARK,
        space_after=14.0,
    )

    # Right Column: 3 Future Directions
    add_rect(slide, 6.833, 1.26, 5.85, 4.65, fill_color=C_DARK_CARD, border_color=C_DARK_BORDER, border_pt=1.0, rounded=True)
    add_rect(slide, 6.833, 1.26, 5.85, 0.07, fill_color=C_BLUE_BRIGHT)
    add_textbox(
        slide, 7.08, 1.46, 5.35, 0.34, text="Future Research Directions", font_size=15.0, bold=True, color=C_BLUE_BRIGHT, margin_lr=0.0, margin_tb=0.0
    )
    add_bullets(
        slide,
        7.08,
        1.94,
        5.35,
        3.80,
        [
            ("1. Test-Time BatchNorm Adaptation: ", "Update running mean/variance statistics on unlabeled target batches to counter baseline decay."),
            ("2. Unsupervised Domain Adaptation: ", "Apply Domain-Adversarial Neural Networks (DANN) or CORAL alignment across consecutive batches."),
            ("3. Recency-Weighted Ensembles: ", "Prioritize recent calibration batches over stale early data and handle returning classes like Toluene."),
        ],
        font_size=11.5,
        lead_color=C_TEXT_LIGHT,
        body_color=C_TEXT_MUTED_DARK,
        space_after=14.0,
    )

    # Bottom Q&A banner
    add_rect(slide, 0.65, 6.18, 12.033, 0.80, fill_color=C_DARK_CARD, border_color=C_TEAL_BRIGHT, border_pt=1.0, rounded=True)
    add_textbox(
        slide,
        0.90,
        6.26,
        11.53,
        0.64,
        text="Thank you for your attention!  ·  Questions & Discussion\nCode, Notebooks & Technical Report: https://github.com/z64x/Deeplearn-project",
        font_size=12.0,
        bold=True,
        color=C_TEXT_LIGHT,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin_lr=0.0,
        margin_tb=0.0,
    )

    set_notes(
        slide,
        "Slide 14 (~1:00): În concluzie: validarea cronologică este indispensabilă pe senzori cu drift; MLP-ul pe toate cele "
        "128 de caracteristici oferă cel mai stabil compromis sub drift pe 36 de luni (82.57% medie / 80.31% pooled), în timp ce "
        "UMAP out-of-sample degradează performanța. Ca direcții viitoare: adaptarea statisticilor BatchNorm la test-time, "
        "Domain Adaptation (DANN/CORAL) și ansambluri ponderate temporal. Mulțumesc și aștept întrebările!",
    )


def build_presentation() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    build_slide_01_title(prs)
    build_slide_02_problem(prs)
    build_slide_03_dataset(prs)
    build_slide_04_sensor_drift(prs)
    build_slide_05_pca(prs)
    build_slide_06_architectures(prs)
    build_slide_07_validation(prs)
    build_slide_08_umap(prs)
    build_slide_09_random_vs_temporal(prs)
    build_slide_10_benchmark_table(prs)
    build_slide_11_per_batch(prs)
    build_slide_12_confusion(prs)
    build_slide_13_software(prs)
    build_slide_14_conclusions(prs)

    OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PPTX))
    print(f"Saved presentation ({len(prs.slides)} slides) to: {OUTPUT_PPTX}")
    return OUTPUT_PPTX


if __name__ == "__main__":
    build_presentation()
