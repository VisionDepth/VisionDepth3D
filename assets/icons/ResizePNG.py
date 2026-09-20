from PIL import Image
from pathlib import Path

SOURCE = "logo.png"
OUTPUT = "app_icon.ico"

sizes = [16, 24, 32, 48, 64, 128, 256]

src = Image.open(SOURCE).convert("RGBA")

# Create a clean 1024x1024 master canvas first
master_size = 1024
padding = 120

canvas = Image.new("RGBA", (master_size, master_size), (0, 0, 0, 0))

# Preserve aspect ratio
logo = src.copy()
logo.thumbnail((master_size - padding * 2, master_size - padding * 2), Image.Resampling.LANCZOS)

x = (master_size - logo.width) // 2
y = (master_size - logo.height) // 2
canvas.paste(logo, (x, y), logo)

# Save PNG preview so you can inspect it
canvas.save("app_icon_master.png")

# Save proper multi-size ICO
canvas.save(
    OUTPUT,
    format="ICO",
    sizes=[(s, s) for s in sizes]
)

print(f"Saved {OUTPUT}")
print("Also saved app_icon_master.png for preview")