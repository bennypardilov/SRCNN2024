from utils.common import *
from model import SRCNN
from neuralnet import SRCNN_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int,   default=2,                   help='-')
parser.add_argument("--ckpt-path",    type=str,   default="",                  help='-')
parser.add_argument("--image-path",   type=str,   default="dataset/test1.png", help='-')

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("must be 2, 3 or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNNckpt.pt"

sigma = 0.3 if scale == 2 else 0.2



# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose an LR image from the training set
    lr_image = read_image(image_path).to(DEVICE)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = rgb2ycbcr(bicubic_image)
    bicubic_image = norm01(bicubic_image)
    bicubic_image = torch.unsqueeze(bicubic_image, dim=0)

    model = SRCNN_model(3, 3).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    with torch.no_grad():
        bicubic_image = bicubic_image.to(DEVICE)
        sr_output = model(bicubic_image.unsqueeze(0)).cpu()  # Unsqueeze to add batch dimension and move to CPU

    # Convert tensors to numpy arrays and remove batch dimension
    lr_image = lr_image.permute(1, 2, 0).cpu().numpy() + 0.5
    sr_output = sr_output.squeeze().permute(1, 2, 0).numpy() + 0.5

    sr_output = denorm01(sr_output)
    sr_output = sr_output.type(torch.uint8)
    sr_output = ycbcr2rgb(sr_output)

    write_image("sr.png", sr_output)

if __name__ == "__main__":
    main()