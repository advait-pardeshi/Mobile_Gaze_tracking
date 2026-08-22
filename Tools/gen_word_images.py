#!/usr/bin/env python3
"""Generate the word-tile PNGs for Experiment 3 (communication task).

One tile per word: white card, large bold black word, and a coloured accent
bar along the top so the tiles read as distinct cards at a glance without
relying on the text alone.

Target words and distractors are tinted differently ONLY in hue-neutral terms
(all accents are mid-saturation); the colour is not a cue to which words are
in the target sentence, because that would let a participant shortcut the
search task.

Output basenames follow `CommunicationWordSet.imageName`: word_<lowercase>.png
"""

import os

from PIL import Image, ImageDraw, ImageFont

# Resources/Experiment3Images, resolved relative to this file so the script
# works from any working directory.
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Resources", "Experiment3Images")

# Rendered at 3x the on-screen tile size so the images stay crisp on a
# Retina display without being needlessly large.
W, H = 420, 300
ACCENT_H = 26

WORDS = [
    "I", "want", "to", "drink", "water",
    "eat", "sleep", "help", "more", "please", "stop", "home",
]

# A fixed palette cycled by position in the list, deliberately unrelated to
# whether a word is part of the target sentence.
ACCENTS = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (92, 107, 192),
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fitted_font(draw, text, max_w, max_h, start=150):
    """Largest font size whose rendered text fits the given box."""
    size = start
    while size > 12:
        font = load_font(size)
        box = draw.textbbox((0, 0), text, font=font)
        if (box[2] - box[0]) <= max_w and (box[3] - box[1]) <= max_h:
            return font
        size -= 4
    return load_font(12)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for i, word in enumerate(WORDS):
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, W, ACCENT_H], fill=ACCENTS[i % len(ACCENTS)])

        font = fitted_font(draw, word, max_w=W - 110, max_h=H - ACCENT_H - 110)
        # `anchor="mm"` centres on the glyph box using the font's real ascent
        # and descent, so a word with a descender ("please") sits at the same
        # optical centre as one without ("water"). Manual bbox arithmetic
        # centres the *ink* instead, which makes descender words ride low.
        cx = W / 2
        cy = ACCENT_H + (H - ACCENT_H) / 2
        draw.text((cx, cy), word, font=font, fill=(17, 17, 17), anchor="mm")

        name = "word_" + word.lower().replace(" ", "_") + ".png"
        img.save(os.path.join(OUT_DIR, name), "PNG", optimize=True)
        written.append(name)

    print(f"wrote {len(written)} tiles to {OUT_DIR}")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
