import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import PIL

def negative_op(img, extra_arguments=None):
    return 255 - img

def interval_transform_op(img,in_min=0, in_max=255, out_min=100, out_max=200):
    return ((out_max - out_min)/(in_max - in_min))*(img-in_min) + out_min

def wrong_mode(img, extra_arguments=None):
    print('Inserted mode does not exist')
    return img

def gamma_correction(img, gamma = 3.5):
    # First convert it to the interval [0,1]
    img_tmp = interval_transform_op(img,0,255,0,1)
    img_tmp = img_tmp**float(1/float(gamma))
    img_tmp = interval_transform_op(img_tmp,0,1,0,255)
    return img_tmp

def quantization(img, colors = 2):
    # First transformto negative to match the quantize() method output
    pil_img = Image.fromarray(negative_op(img))
    pil_img = pil_img.quantize(int(colors))
    return np.array(pil_img)

def bit_plan(img, plan_n):
    mask = 2**int(plan_n)
    return img & mask

def combine_images(img1, img2, prop1, prop2):
    img2 = mpimg.imread(img2)[:,:,0]
    prop1 = float(prop1)
    prop2 = float(prop2)

    if img1.shape != img2.shape:
        # make the size of both images equal
        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((img1.shape[1],img1.shape[0]))
        # Convert it back to np array
        img2 = np.array(img2_pil)

    if prop1 + prop2 > 1:
        print('Proportions not adequate, image may lose info')

    return img1*prop1 + img2*prop2

def operate_image(mode, input_img_path, extra_arguments, output_img_path = 'output.png'):
    # Dictionary that selects which function will be called depending on the inserted mode
    operations = {
    'negative': negative_op,
    'transform-interval': interval_transform_op,
    'gamma-correction': gamma_correction,
    'quantization': quantization,
    'bit-plan': bit_plan,
    'combine-images': combine_images
    }
    input_img = mpimg.imread(input_img_path)[:,:,0]
    output_img = operations.get(mode, wrong_mode)(input_img, *extra_arguments)

    # Quantization and bit_plan mode require the vmin and vmax arguments to not be set
    if mode == 'quantization' or mode == 'bit-plan':
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 255

    plt.axis('off')
    plt.imshow(output_img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.savefig(output_img_path)
    plt.show()

if __name__ == '__main__':
    arg_count = 2
    input_img_path = sys.argv[1]
    output_img_path = 'output.png'
    if sys.argv[2] == '-o':
        # has output file name
        arg_count += 1
        output_img_path = sys.argv[3]
        arg_count += 1
        mode = sys.argv[4]
        arg_count += 1
        print(f'out é {output_img_path}')
    else:
        mode = sys.argv[2]
        arg_count += 1
    if len(sys.argv) > arg_count:
        # Has a output_path input
        extra_arguments = sys.argv[arg_count:]
        operate_image(mode, input_img_path, extra_arguments, output_img_path)
    else:
        extra_arguments = []
        operate_image(mode, input_img_path, extra_arguments, output_img_path)
