import boto3
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def detect_faces_rekognition(image_bytes):
    client = boto3.client('rekognition')
    response = client.detect_faces(Image={'Bytes': image_bytes}, Attributes=['ALL'])
    return response['FaceDetails']

def get_center_face(faces, frame_width, frame_height):
    if len(faces) == 0:
        return None
    
    frame_center = (frame_width // 2, frame_height // 2)
    
    def distance_to_center(face):
        box = face['BoundingBox']
        face_center = (
            int((box['Left'] + box['Width'] / 2) * frame_width),
            int((box['Top'] + box['Height'] / 2) * frame_height)
        )
        return np.sqrt((face_center[0] - frame_center[0])**2 + (face_center[1] - frame_center[1])**2)
    
    return min(faces, key=distance_to_center)

def display_custom_labels(image, response):
    draw = ImageDraw.Draw(image)
    imgWidth, imgHeight = image.size

    for customLabel in response['CustomLabels']:
        print('Label ' + str(customLabel['Name']))
        print('Confidence ' + str(customLabel['Confidence']))
        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            fnt = ImageFont.load_default()
            draw.text((left, top), customLabel['Name'], fill='#00d400', font=fnt)

            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top)
            )
            draw.line(points, fill='#00d400', width=2)

    return image

def detect_custom_labels(model, image_bytes, min_confidence):
    client = boto3.client('rekognition')

    try:
        response = client.detect_custom_labels(
            Image={'Bytes': image_bytes},
            MinConfidence=min_confidence,
            ProjectVersionArn=model
        )
        return response
    except client.exceptions.InvalidImageFormatException:
        print("Invalid image format. Skipping this face.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def process_video(video_path, model, min_confidence, output_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Convert frame to bytes for Rekognition
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        faces = detect_faces_rekognition(image_bytes)
        center_face = get_center_face(faces, width, height)
        
        if center_face is not None:
            box = center_face['BoundingBox']
            x = int(box['Left'] * width)
            y = int(box['Top'] * height)
            w = int(box['Width'] * width)
            h = int(box['Height'] * height)
            
            face_region = frame[y:y+h, x:x+w]
            
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            
            # Save as JPEG
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            face_bytes = buffer.getvalue()
            
            response = detect_custom_labels(model, face_bytes, min_confidence)
            
            if response and 'CustomLabels' in response:
                pil_image = display_custom_labels(pil_image, response)
                
                # Convert back to OpenCV format and place in the frame
                face_region_with_labels = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                frame[y:y+h, x:x+w] = face_region_with_labels

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def main():
    video_path = './temp/sample.mp4'
    model = 'arn:model arn'
    min_confidence = 50
    output_path = './temp/Sample.mp4'

    process_video(video_path, model, min_confidence, output_path)

if __name__ == "__main__":
    main()
