import os
import cv2
import json
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm  # For progress bar
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv("API_KEY")
API_URL = "https://api.openalpr.com/v1/recognize?tasks=plate&recognize_vehicle=0&country=us&secret_key=" + API_KEY

def send_image_to_api(image_path):
    with open(image_path, 'rb') as image_file:
        response = requests.post(API_URL, files={'image': image_file})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def parse_ocr_data(ocr_data):
    results = ocr_data.get('plate', {}).get('results', [])
    if not results:
        return []

    plates = []
    for result in results:
        plate_data = {
            'plate': result['plate'],
            'confidence': result['confidence'],
            'coordinates': result['coordinates']
        }
        plates.append(plate_data)
    return plates

def draw_box_and_save(image_path, plate_data, frame_name=None):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    coordinates = plate_data["coordinates"]
    box = [(coord["x"], coord["y"]) for coord in coordinates]
    outline_width = 5
    draw.polygon(box, outline="black", width=outline_width)

    plate_text = f"{plate_data['plate']} ({plate_data['confidence']:.2f}%)"
    text_position = (coordinates[0]["x"], coordinates[0]["y"] - 30)
    draw.text(text_position, plate_text, fill="black", font=font)

    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    output_filename = f"{plate_data['plate']}.jpg" if frame_name is None else f"{frame_name}_{plate_data['plate']}.jpg"
    output_path = os.path.join(results_folder, output_filename)
    image.save(output_path)
    #print(f"Saved annotated image as: {output_path}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, fps // 2)  # Capture frames every half-second

    plates_found = []

    with tqdm(total=frame_count // interval, desc="Processing Video", unit="frames") as pbar:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % interval == 0:
                frame_path = f"temp_frame_{frame_number}.jpg"
                cv2.imwrite(frame_path, frame)
                ocr_data = send_image_to_api(frame_path)
                
                if ocr_data:
                    detected_plates = parse_ocr_data(ocr_data)
                    for plate_data in detected_plates:
                        draw_box_and_save(frame_path, plate_data, frame_name=f"frame_{frame_number}")
                        plates_found.append({
                            'plate': plate_data['plate'],
                            'confidence': plate_data['confidence'],
                            'frame': frame_number
                        })

                os.remove(frame_path)  # Remove temp file after saving annotated image
                pbar.update(1)

            frame_number += 1

    cap.release()
    print("\nVideo processing complete.")
    return plates_found

def main():
    while True:
        print("Choose an option:")
        print("1. Process an image")
        print("2. Process a video")
        choice = input("Enter your choice (1 or 2, or 'exit' to quit): ").strip()

        if choice == '1':
            image_folder = "img"
            images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                print("No images found in the img folder.")
                continue

            print("\nAvailable Images:")
            for i, image_name in enumerate(images, start=1):
                print(f"{i}. {image_name}")
            selection = input("Please choose an image to process (or type 'exit' to quit): ")

            if selection.lower() == 'exit':
                break

            try:
                image_choice = int(selection) - 1
                if image_choice not in range(len(images)):
                    raise ValueError("Invalid selection.")
                image_path = os.path.join(image_folder, images[image_choice])

                ocr_data = send_image_to_api(image_path)
                if ocr_data:
                    detected_plates = parse_ocr_data(ocr_data)
                    if detected_plates:
                        print("\nDetected Plates:")
                        for i, plate_data in enumerate(detected_plates, start=1):
                            print(f"{i}. {plate_data['plate']}")
                        choice = input("\nEnter the number of the plate to see more info, or 'skip' to go back: ")
                        if choice.isdigit() and 1 <= int(choice) <= len(detected_plates):
                            plate_data = detected_plates[int(choice) - 1]
                            print("\nDetailed Plate Information:")
                            print(f"Plate: {plate_data['plate']}")
                            print(f"Confidence: {plate_data['confidence']}")
                            print("Bounding Box Coordinates:")
                            for coord in plate_data['coordinates']:
                                print(f"  - x: {coord['x']}, y: {coord['y']}")
                            draw_box_and_save(image_path, plate_data)
                        else:
                            print("Skipping to next image.")
                    else:
                        print("No plates detected.")
                else:
                    print("Error in OCR processing.")

            except ValueError:
                print("Invalid input. Please enter a number corresponding to the image.")

        elif choice == '2':
            video_folder = "vid"
            videos = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
            if not videos:
                print("No videos found in the vid folder.")
                continue

            print("\nAvailable Videos:")
            for i, video_name in enumerate(videos, start=1):
                print(f"{i}. {video_name}")
            selection = input("Please choose a video to process (or type 'exit' to quit): ")

            if selection.lower() == 'exit':
                break

            try:
                video_choice = int(selection) - 1
                if video_choice not in range(len(videos)):
                    raise ValueError("Invalid selection.")
                video_path = os.path.join(video_folder, videos[video_choice])

                plates_found = process_video(video_path)
                if plates_found:
                    print("\nAll Detected Plates:")
                    for i, plate_info in enumerate(plates_found, start=1):
                        print(f"{i}. Plate: {plate_info['plate']} (Confidence: {plate_info['confidence']:.2f}%) - Frame: {plate_info['frame']}")

                    # Loop to view multiple plates
                    while True:
                        choice = input("\nEnter the number of a plate to view more details, or type 'done' to continue: ").strip()
                        if choice.lower() == 'done':
                            break
                        if choice.isdigit() and 1 <= int(choice) <= len(plates_found):
                            plate_info = plates_found[int(choice) - 1]
                            print("\nDetailed Plate Information:")
                            print(f"Plate: {plate_info['plate']}")
                            print(f"Confidence: {plate_info['confidence']:.2f}%")
                            print(f"Frame: {plate_info['frame']}")
                        else:
                            print("Invalid input. Please enter a valid plate number or 'done'.")

                else:
                    print("No plates detected in the video.")

            except ValueError:
                print("Invalid input. Please enter a number corresponding to the video.")

        elif choice.lower() == 'exit':
            break
        else:
            print("Invalid choice. Please enter '1', '2', or 'exit'.")

    print("Exiting.")

if __name__ == "__main__":
    main()
