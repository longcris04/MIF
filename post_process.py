import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob


def is_grayscale(image):
    """Check if an image is grayscale"""
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3:
        # Check if all channels are the same
        if image.shape[2] == 1:
            return True
        # Check if R, G, B channels are identical
        return np.allclose(image[:, :, 0], image[:, :, 1]) and np.allclose(image[:, :, 1], image[:, :, 2])
    return False


def enhance_brightness(image, alpha=1.2, beta=30):
    """
    Enhance brightness of the image
    alpha: contrast control (1.0-3.0)
    beta: brightness control (0-100)
    """
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced


def enhance_contrast_histogram(image):
    """
    Enhance contrast using histogram equalization
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced


def sharpen_image(image, kernel_type='laplacian'):
    """
    Enhance image details using sharpening filters
    kernel_type: 'laplacian', 'unsharp_mask', or 'kernel'
    """
    if kernel_type == 'laplacian':
        # Laplacian sharpening
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = cv2.convertScaleAbs(image - laplacian * 0.5)
    elif kernel_type == 'unsharp_mask':
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    else:
        # Custom sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def apply_all_enhancements(image):
    """
    Apply all enhancement techniques sequentially
    """
    enhanced = image.copy()
    # 1. Enhance brightness
    enhanced = enhance_brightness(enhanced, alpha=1.2, beta=20)
    
    # 2. Enhance contrast with histogram equalization
    # enhanced = enhance_contrast_histogram(enhanced)
    
    # 3. Sharpen image details
    # enhanced = sharpen_image(enhanced, kernel_type='unsharp_mask')
    # enhanced = sharpen_image(enhanced, kernel_type='laplacian')
    
    return enhanced


def process_rgb_image(image):
    """
    Process RGB image by converting to YCbCr, processing Y channel, and merging back
    """
    # Convert BGR to YCrCb (OpenCV uses YCrCb instead of YCbCr)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split channels
    y, cr, cb = cv2.split(ycrcb)
    
    # Apply enhancements to Y channel only
    y_enhanced = apply_all_enhancements(y)
    
    # Merge channels back
    ycrcb_enhanced = cv2.merge([y_enhanced, cr, cb])
    
    # Convert back to BGR
    rgb_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)
    
    return rgb_enhanced


def process_grayscale_image(image):
    """
    Process grayscale image directly
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply all enhancements
    enhanced = apply_all_enhancements(gray)
    
    return enhanced


def process_image(input_path, output_path, channel='rgb'):
    """
    Process a single image
    """
    # Read image
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Error: Cannot read image {input_path}")
        return False
    
    # Check if image is grayscale
    is_gray = is_grayscale(image)
    
    # Process based on channel requirement and image type
    if channel == 'gray':
        # Output as grayscale
        processed = process_grayscale_image(image)
    else:
        # Output as RGB
        if is_gray:
            # Convert grayscale to RGB first
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            processed = process_rgb_image(image)
        else:
            processed = process_rgb_image(image)
    
    # Save processed image
    cv2.imwrite(output_path, processed)
    return True


def main():
    parser = argparse.ArgumentParser(description='Post-process medical image fusion results')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory or image path containing .bmp files')
    parser.add_argument('-s','--savedir', type=str, default='./post_processed_results',
                        help='Base directory to save post-processed results')
    parser.add_argument('-n','--method_name', type=str, required=True,
                        help='Method name for organizing results')
    parser.add_argument('-m','--modality', type=str, required=True,
                        help='Modality pair (e.g., CT-MRI, PET-MRI, SPECT-MRI)')
    parser.add_argument('-c','--channel', type=str, choices=['gray', 'rgb'], default='rgb',
                        help='Output channel type: gray or rgb')
    
    args = parser.parse_args()
    
    # Create output directory structure
    if args.channel == 'gray':
        output_dir = os.path.join(args.savedir, args.method_name, args.modality, 'results_gray')
    else:
        output_dir = os.path.join(args.savedir, args.method_name, args.modality, 'results_rgb')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get list of input images
    if os.path.isdir(args.input):
        # Process all .bmp files in directory
        image_paths = glob.glob(os.path.join(args.input, '*.bmp'))
    elif os.path.isfile(args.input):
        # Process single image
        image_paths = [args.input]
    else:
        print(f"Error: Input path {args.input} does not exist")
        return
    
    if len(image_paths) == 0:
        print(f"No .bmp images found in {args.input}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    success_count = 0
    for img_path in image_paths:
        # Get filename
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing: {filename}...", end=' ')
        
        if process_image(img_path, output_path, args.channel):
            success_count += 1
            print("Done")
        else:
            print("Failed")
    
    print(f"\nProcessing complete: {success_count}/{len(image_paths)} images processed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
