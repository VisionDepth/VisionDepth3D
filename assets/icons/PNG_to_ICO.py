from PIL import Image
from pathlib import Path

SOURCE = "app_icon_master.png"
OUTPUT = "logo.ico"

sizes = [256, 128, 64, 48, 32, 24, 16]

src_path = Path(SOURCE)
if not src_path.exists():
    raise FileNotFoundError(f"Could not find {SOURCE}")

img = Image.open(src_path).convert("RGBA")

print("Source image size:", img.size)

# Crop transparent padding around the actual logo.
alpha = img.getchannel("A")
bbox = alpha.getbbox()

if bbox:
    img = img.crop(bbox)
    print("Cropped visible logo size:", img.size)
else:
    print("No alpha bbox found. Using full image.")

# Put the cropped logo on a square canvas.
visible_size = max(img.width, img.height)

# Padding percentage.
# Lower = bigger logo.
# 0.08 means 8% padding around the logo.
padding_ratio = 0.08
canvas_size = int(visible_size / (1.0 - padding_ratio * 2))

canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

x = (canvas_size - img.width) // 2
y = (canvas_size - img.height) // 2
canvas.paste(img, (x, y), img)

icon_images = []

for size in sizes:
    resized = canvas.resize((size, size), Image.Resampling.LANCZOS)
    icon_images.append(resized)
    resized.save(f"icon_{size}x{size}.png")

icon_images[0].save(
    OUTPUT,
    format="ICO",
    sizes=[(s, s) for s in sizes],
    append_images=icon_images[1:],
)

print(f"Saved {OUTPUT}")

ico = Image.open(OUTPUT)
print("ICO embedded sizes:", ico.ico.sizes())