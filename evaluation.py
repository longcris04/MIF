"""
Medical Image Fusion Evaluation Script
This script runs MATLAB evaluation code for medical image fusion results.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_matlab_evaluation(path_1, path_2, path_3, member, output_folder=None, method='engine'):
    """
    Run MATLAB evaluation script with specified parameters.
    
    Args:
        path_1: Path to source image 1 (e.g., CT, PET, SPECT)
        path_2: Path to source image 2 (e.g., MRI)
        path_3: Path to fused images
        member: Dataset type (e.g., 'CT-MRI', 'PET-MRI', 'SPECT-MRI')
        output_folder: Output folder for results (default: same as path_3 parent)
        method: 'engine' for MATLAB Engine API or 'subprocess' for command line
    """
    
    # Validate paths
    if not os.path.exists(path_1):
        raise ValueError(f"path_1 does not exist: {path_1}")
    if not os.path.exists(path_2):
        raise ValueError(f"path_2 does not exist: {path_2}")
    if not os.path.exists(path_3):
        raise ValueError(f"path_3 does not exist: {path_3}")
    
    # Set output folder
    if output_folder is None:
        output_folder = str(Path(path_3).parent)
    
    # Get the matlab_evaluation_code directory
    script_dir = Path(__file__).parent
    matlab_code_dir = script_dir / "matlab_evaluation_code"
    
    if not matlab_code_dir.exists():
        raise ValueError(f"MATLAB code directory not found: {matlab_code_dir}")
    
    print(f"Running MATLAB evaluation...")
    print(f"  Source 1: {path_1}")
    print(f"  Source 2: {path_2}")
    print(f"  Fused:    {path_3}")
    print(f"  Member:   {member}")
    print(f"  Output:   {output_folder}")
    print()
    
    if method == 'engine':
        run_with_matlab_engine(path_1, path_2, path_3, member, output_folder, matlab_code_dir)
    else:
        run_with_subprocess(path_1, path_2, path_3, member, output_folder, matlab_code_dir)


def run_with_matlab_engine(path_1, path_2, path_3, member, output_folder, matlab_code_dir):
    """Run MATLAB using the MATLAB Engine API for Python."""
    try:
        import matlab.engine
    except ImportError:
        print("ERROR: MATLAB Engine API for Python is not installed.")
        print("Please install it or use --method subprocess instead.")
        print("\nTo install MATLAB Engine API:")
        print("  cd \"matlabroot\\extern\\engines\\python\"")
        print("  python setup.py install")
        sys.exit(1)
    
    print("Starting MATLAB Engine...")
    eng = matlab.engine.start_matlab()
    
    try:
        # Change to MATLAB code directory
        eng.cd(str(matlab_code_dir), nargout=0)
        
        # Set the variables in MATLAB workspace
        eng.workspace['path_1'] = path_1
        eng.workspace['path_2'] = path_2
        eng.workspace['path_3'] = path_3
        eng.workspace['member'] = member
        eng.workspace['outputFolder'] = output_folder
        
        # Create a modified version of the script that uses the variables
        print("Running MATLAB evaluation script...")
        
        # Run the modified script
        eng.eval("""
            close all; clear path_1 path_2 path_3 member outputFolder;
            addpath(genpath('indexes'));
            addpath(genpath(cd));
        """, nargout=0)
        
        # Set variables again after clear
        eng.workspace['path_1'] = path_1
        eng.workspace['path_2'] = path_2
        eng.workspace['path_3'] = path_3
        eng.workspace['member'] = member
        eng.workspace['custom_output'] = output_folder
        
        # Run the main evaluation code
        eng.eval("""
            if ~exist(custom_output, 'dir')
                mkdir(custom_output);
            end
            outputFolder = custom_output;
            
            % Define text file for results
            filename_full = append(member, '_full.txt');
            path_full = fullfile(outputFolder, filename_full);
            fileID_slice = fopen(path_full, 'w');
            
            % Define headers
            headers = {'Num', 'Name', 'MLI', 'SD', 'Entropy', 'AG', 'Qabf', 'VIFF', 'FMI', 'MI', ...
                       'Xydeas', 'Qp', 'Q', 'Qw', 'Qe', 'NIQE', 'SSEQ', 'FSIM', 'SSIM', 'TMQI', ...
                       'PSNR', 'Chen', 'CB', 'metricWang', 'NCIE', 'metricZhao', 'x1', 'x2', 'Time'};
            
            headerFormat = '%-5s %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\\n';
            fprintf(fileID_slice, headerFormat, headers{:});
            
            % Initialize arrays
            name_f = {};
            v1_f = [];
            y_f = [];
            time_f = [];
            
            % Read image files
            original_files_1 = dir(fullfile(path_1, '*.bmp'));
            original_files_2 = dir(fullfile(path_2, '*.bmp'));
            original_files_3 = dir(fullfile(path_3, '*.bmp'));
            
            if length(original_files_1) == length(original_files_2)
                for k = 1:length(original_files_1)
                    fprintf('Processing image %d/%d...\\n', k, length(original_files_1));
                    filename1 = fullfile(path_1, original_files_1(k).name);
                    filename2 = fullfile(path_2, original_files_2(k).name);
                    filename3 = fullfile(path_3, original_files_3(k).name);
                    [~, baseFileNameNoExt1, ~] = fileparts(filename1);
                    [~, baseFileNameNoExt2, ~] = fileparts(filename2);
                    
                    if strcmp(baseFileNameNoExt1, baseFileNameNoExt2)
                        A = im2double(imread(filename1));
                        img2 = im2double(imread(filename2));
                        imgf = im2double(imread(filename3));
                        
                        if size(A, 3) > 1
                            A = rgb2gray(A);
                        end
                        
                        if size(img2, 3) > 1
                            B_YUV = ConvertRGBtoYUV(img2);
                            BB = B_YUV(:, :, 1);
                        else
                            BB = img2;
                        end
                        
                        tic
                        time = toc;
                        if size(imgf, 3) > 1
                            F = rgb2gray(imgf);
                        else
                            F = imgf;
                        end
                        
                        % Evaluate Metrics
                        v = DanhGiaAnh(F);
                        v1 = zeros(1,4);
                        v1(1) = v(1);
                        v1(2) = v(2);
                        v1(3) = v(3);
                        v1(4) = v(4);
                        
                        im1 = im2uint8(A);
                        im2 = im2uint8(BB);
                        imf = im2uint8(F);
                        
                        y = zeros(1, 22);
                        y(1) = Qabf(im1, im2, imf);
                        y(2) = VIFF(im1, im2, imf);
                        y(3) = fmi(im1, im2, imf, 'edge', 3);
                        y(4) = mutual_information(im1, im2, imf, 256);
                        y(5) = metricXydeas(im1, im2, imf);
                        y(6) = Qp_ABF(im1, im2, imf);
                        y(7) = metricPeilla(im1, im2, imf, 1);
                        y(8) = metricPeilla(im1, im2, imf, 2);
                        y(9) = metricPeilla(im1, im2, imf, 3);
                        y(10) = niqe(imf);
                        y(11) = SSEQ(cat(3, im1, im2, imf));
                        y(12) = FSIM(im1, imf);
                        y(13) = MS_SSIM(im1, im2, imf);
                        y(14) = TMQI(cat(3,im1, im1, im1), cat(3,imf, imf, imf));
                        y(15) = psnr(imf,im1);
                        y(16) = metricChen(im1, im2, imf);
                        y(17) = metricChenBlum(im1, im2, imf);
                        y(18) = metricWang(im1, im2, imf);
                        y(19) = Q_NCIE(im1,im2,imf);
                        y(20) = -1;
                        
                        name_f = [name_f; {baseFileNameNoExt1}];
                        v1_f = [v1_f; v1];
                        y_f = [y_f; y];
                        time_f = [time_f; time];
                        
                        rowFormat = '%-5d %-15s %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\\n';
                        fprintf(fileID_slice, rowFormat, k, baseFileNameNoExt1, v1, y, time);
                    end
                end
            end
            
            % Calculate statistics
            numEntries = min([length(v1_f), size(y_f, 1), length(time_f)]);
            v1_f = v1_f(1:numEntries, :);
            y_f = y_f(1:numEntries, :);
            time_f = time_f(1:numEntries);
            
            v1_max = max(v1_f, [], 1);
            y_max = max(y_f, [], 1);
            time_max = max(time_f);
            
            v1_min = min(v1_f, [], 1);
            y_min = min(y_f, [], 1);
            time_min = min(time_f);
            
            v1_mean = mean(v1_f, 1);
            y_mean = mean(y_f, 1);
            time_mean = mean(time_f);
            
            v1_std = std(v1_f, 1);
            y_std = std(y_f, 1);
            time_std = std(time_f);
            
            rowIndices = num2cell((1:numEntries)');
            baseNames = name_f;
            v1_f_cells = num2cell(v1_f);
            y_f_cells = num2cell(y_f);
            time_f_cells = num2cell(time_f);
            
            excelData = [rowIndices, baseNames, v1_f_cells, y_f_cells, time_f_cells];
            
            excelData = [excelData; ...
                {[], 'Max', v1_max(1), v1_max(2), v1_max(3), v1_max(4), y_max(1), y_max(2), y_max(3), y_max(4), y_max(5), y_max(6), y_max(7), y_max(8), y_max(9), y_max(10), y_max(11), y_max(12), y_max(13), y_max(14), y_max(15), y_max(16), y_max(17), y_max(18), y_max(19), y_max(20), y_max(21), y_max(22), time_max}; ...
                {[], 'Min', v1_min(1), v1_min(2), v1_min(3), v1_min(4), y_min(1), y_min(2), y_min(3), y_min(4), y_min(5), y_min(6), y_min(7), y_min(8), y_min(9), y_min(10), y_min(11), y_min(12), y_min(13), y_min(14), y_min(15), y_min(16), y_min(17), y_min(18), y_min(19), y_min(20), y_min(21), y_min(22), time_min}; ...
                {[], 'Mean', v1_mean(1), v1_mean(2), v1_mean(3), v1_mean(4), y_mean(1), y_mean(2), y_mean(3), y_mean(4), y_mean(5), y_mean(6), y_mean(7), y_mean(8), y_mean(9), y_mean(10), y_mean(11), y_mean(12), y_mean(13), y_mean(14), y_mean(15), y_mean(16), y_mean(17), y_mean(18), y_mean(19), y_mean(20), y_mean(21), y_mean(22), time_mean}; ...
                {[], 'Std', v1_std(1), v1_std(2), v1_std(3), v1_std(4), y_std(1), y_std(2), y_std(3), y_std(4), y_std(5), y_std(6), y_std(7), y_std(8), y_std(9), y_std(10), y_std(11), y_std(12), y_std(13), y_std(14), y_std(15), y_std(16), y_std(17), y_std(18), y_std(19), y_std(20), y_std(21), y_std(22), time_std}];
            
            currentFolder = pwd;
            [~, currentFolderName] = fileparts(currentFolder);
            outputExcelFile = fullfile(outputFolder, strcat(currentFolderName, '_', member, '_Evaluation.xlsx'));
            
            xlswrite(outputExcelFile, [headers; excelData], 'Data');
            
            summaryFormat = '%-5s %-15s %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\\n';
            fprintf(fileID_slice, summaryFormat, 'Max', '', v1_max, y_max, time_max);
            fprintf(fileID_slice, summaryFormat, 'Min', '', v1_min, y_min, time_min);
            fprintf(fileID_slice, summaryFormat, 'Mean', '', v1_mean, y_mean, time_mean);
            fprintf(fileID_slice, summaryFormat, 'Std', '',  v1_std, y_std, time_std);
            
            fclose(fileID_slice);
            
            disp('Data and summary statistics saved to Excel and text file successfully.');
        """, nargout=0)
        
        print("\n✓ MATLAB evaluation completed successfully!")
        print(f"Results saved to: {output_folder}")
        
    finally:
        eng.quit()


def run_with_subprocess(path_1, path_2, path_3, member, output_folder, matlab_code_dir):
    """Run MATLAB using subprocess and command line."""
    
    # Create a temporary MATLAB script with the parameters
    temp_script = matlab_code_dir / "temp_run_evaluation.m"
    
    script_content = f"""
% Auto-generated script
close all; clear all; clc;

addpath(genpath('indexes'));
addpath(genpath(cd));

path_1 = '{path_1.replace(chr(92), chr(92)*2)}';
path_2 = '{path_2.replace(chr(92), chr(92)*2)}';
path_3 = '{path_3.replace(chr(92), chr(92)*2)}';
member = "{member}";
outputFolder = '{output_folder.replace(chr(92), chr(92)*2)}';

% Run the evaluation code
cd('{str(matlab_code_dir).replace(chr(92), chr(92)*2)}');
Run_Foder_Updated;
"""
    
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        # Run MATLAB in batch mode
        matlab_cmd = f'matlab -batch "cd(\'{str(matlab_code_dir).replace(chr(92), chr(92)*2)}\'); temp_run_evaluation"'
        
        print(f"Running MATLAB command: {matlab_cmd}\n")
        
        result = subprocess.run(
            matlab_cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✓ MATLAB evaluation completed successfully!")
            print(f"Results saved to: {output_folder}")
        else:
            print(f"\n✗ MATLAB evaluation failed with return code {result.returncode}")
            sys.exit(1)
            
    finally:
        # Clean up temporary script
        if temp_script.exists():
            temp_script.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Run MATLAB evaluation for medical image fusion results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate CT-MRI fusion results
  python evaluation.py \\
    --path1 "C:\\Users\\Admin\\MIF\\DatasetBMP_update\\CT-MRI\\test\\CT" \\
    --path2 "C:\\Users\\Admin\\MIF\\DatasetBMP_update\\CT-MRI\\test\\MRI" \\
    --path3 "C:\\Users\\Admin\\MIF\\FATFusion\\results_gray\\CT-MRI" \\
    --member "CT-MRI"
  
  # Evaluate PET-MRI fusion results with custom output folder
  python evaluation.py \\
    --path1 "C:\\Users\\Admin\\MIF\\DatasetBMP_update\\PET-MRI\\test\\PET" \\
    --path2 "C:\\Users\\Admin\\MIF\\DatasetBMP_update\\PET-MRI\\test\\MRI" \\
    --path3 "C:\\Users\\Admin\\MIF\\FATFusion\\results_gray\\PET-MRI" \\
    --member "PET-MRI" \\
    --output "C:\\Users\\Admin\\MIF\\evaluation_results"
  
  # Use subprocess method instead of MATLAB Engine API
  python evaluation.py --path1 ... --path2 ... --path3 ... --member "CT-MRI" --method subprocess
        """
    )
    
    parser.add_argument('--path1', type=str, required=True,
                        help='Path to source image 1 (e.g., CT, PET, SPECT)')
    parser.add_argument('--path2', type=str, required=True,
                        help='Path to source image 2 (e.g., MRI)')
    parser.add_argument('--path3', type=str, required=True,
                        help='Path to fused images')
    parser.add_argument('--member', type=str, required=True,
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'],
                        help='Dataset type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder for results (default: parent of path3)')
    parser.add_argument('--method', type=str, default='engine',
                        choices=['engine', 'subprocess'],
                        help='Method to run MATLAB: engine (MATLAB Engine API) or subprocess (command line)')
    
    args = parser.parse_args()
    
    try:
        run_matlab_evaluation(
            path_1=args.path1,
            path_2=args.path2,
            path_3=args.path3,
            member=args.member,
            output_folder=args.output,
            method=args.method
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
