import os
import cv2
import shutil
from tqdm import tqdm

def convert_videos_to_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through folders in the input directory
    for foldername in tqdm(os.listdir(input_dir)):
        folderpath = os.path.join(input_dir, foldername)

        # Check if it is a directory
        if os.path.isdir(folderpath):
            # Create a corresponding folder in the output directory
            output_folderpath = os.path.join(output_dir, foldername)
            os.makedirs(output_folderpath, exist_ok=True)

            # Iterate through files in the folder
            for filename in os.listdir(folderpath):
                filepath = os.path.join(folderpath, filename)

                # Check if the file is an MPEG or MP4 video
                if os.path.isfile(filepath) and (filename.endswith(".mpeg") or filename.endswith(".mp4")):
                    # Create a subfolder with the video name
                    video_name = os.path.splitext(filename)[0]
                    video_output_folderpath = os.path.join(output_folderpath, video_name)
                    os.makedirs(video_output_folderpath, exist_ok=True)

                    video_capture = cv2.VideoCapture(filepath)
                    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                    frame_interval = int(video_capture.get(cv2.CAP_PROP_FPS))

                    frame_count = 0
                    while frame_count<total_frames:
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                        success, frame = video_capture.read()
                        if not success:
                            break

                        # Save the frame as an image
                        frame_filename = f"{video_name}_{frame_count:04d}.jpg"
                        frame_filepath = os.path.join(video_output_folderpath, frame_filename)
                        cv2.imwrite(frame_filepath, frame)

                        frame_count +=  frame_interval

                    # Release the video capture
                    video_capture.release()

                    print(f"Video '{filename}' converted to images and saved in '{video_output_folderpath}'.")

    print("Conversion completed.")

# Example usage
input_directory = "anomaly"
output_directory = "anomaly images"
convert_videos_to_images(input_directory, output_directory)
