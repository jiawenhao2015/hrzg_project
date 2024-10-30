import subprocess
import cv2
import numpy as np

def capture_rtsp_stream(rtsp_url, output_frame_path='output_frame.jpg'):
    # FFmpeg command to capture a single frame from RTSP stream
    ffmpeg_command = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',  # Use TCP transport
        '-i', rtsp_url,  # Input RTSP URL
        '-vframes', '1',  # Capture only one frame
        output_frame_path  # Output file path
    ]

    # Run FFmpeg command
    subprocess.run(ffmpeg_command, check=True)

def main():
    rtsp_url = 'rtsp://admin:rb123456@192.168.66.102:554/Streaming/Channels/101'
    output_frame_path = 'output_frame.jpg'
    
    # Capture frame from RTSP stream
    capture_rtsp_stream(rtsp_url, output_frame_path)
    
    # Load captured frame using OpenCV
    frame = cv2.imread(output_frame_path)
    
    if frame is not None:
        # Display the frame
        cv2.imshow('RTSP Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to capture frame.")

if __name__ == "__main__":
    main()
