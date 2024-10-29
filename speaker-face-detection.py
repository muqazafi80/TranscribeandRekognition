import boto3
import json
import os
from moviepy.editor import VideoFileClip
import requests
import time
import uuid
from collections import defaultdict

def process_video(video_path, bucket_name, s3_key):
    # Initialize AWS clients
    rekognition = boto3.client('rekognition')
    transcribe = boto3.client('transcribe')
    s3 = boto3.client('s3')

    # Extract audio from video
    video = VideoFileClip(video_path)
    audio_path = './temp/audio.wav'
    video.audio.write_audiofile(audio_path)

    # Upload video to S3 if not already there
    if not s3_object_exists(s3, bucket_name, s3_key):
        s3.upload_file(video_path, bucket_name, s3_key)
        print(f"Uploaded video to S3: s3://{bucket_name}/{s3_key}")

    # Perform transcription
    job_name = f"transcribe_job_{int(time.time())}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f"s3://{bucket_name}/{s3_key}"},
        MediaFormat='mp4',
        LanguageCode='en-US',
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 10  # Adjust as needed
        }
    )
    
    # Wait for the transcription job to complete
    print("Waiting for transcription to complete...")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)  # Wait for 5 seconds before checking again
    
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        
        # Download and process the transcript
        transcript_response = requests.get(transcript_uri)
        transcript = json.loads(transcript_response.text)
        
        # Extract speaker segments
        speakers = transcript['results']['speaker_labels']['segments']
        
        # Perform face detection for each speaker segment
        face_to_speaker = detect_faces_in_segments(video_path, speakers, rekognition)
        
        # Process and store the results
        results = process_results(face_to_speaker, transcript)
        
        # Store results locally
        with open('./temp/results.json', 'w') as f:
            json.dump(results, f)
        print("Results stored in ./temp/results.json")
    else:
        print("Transcription job failed")

    # Clean up
    os.remove(audio_path)

def s3_object_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False

def detect_faces_in_segments(video_path, speakers, rekognition):
    face_to_speaker = {}
    speaker_to_face = {}
    video = VideoFileClip(video_path)
    next_face_id = 0

    for speaker in speakers:
        start_time = float(speaker['start_time'])
        end_time = float(speaker['end_time'])
        mid_time = (start_time + end_time) / 2
        speaker_label = speaker['speaker_label']
        
        # If this speaker already has a face assigned, use it
        if speaker_label in speaker_to_face:
            continue
        
        # Extract frame at mid_time
        frame = video.get_frame(mid_time)
        frame_path = f'./temp/frame_{mid_time}.jpg'
        VideoFileClip.save_frame(video, frame_path, t=mid_time)

        # Detect faces in the frame
        with open(frame_path, 'rb') as image:
            response = rekognition.detect_faces(Image={'Bytes': image.read()})

        # Associate detected faces with the speaker
        for detected_face in response['FaceDetails']:
            face_id = f"face_{next_face_id}"
            next_face_id += 1
            
            face_to_speaker[face_id] = {
                'speaker_label': speaker_label,
                'bounding_box': detected_face['BoundingBox'],
                'confidence': detected_face['Confidence']
            }
            speaker_to_face[speaker_label] = face_id
            break  # Only assign one face per speaker

        os.remove(frame_path)

    video.close()
    return face_to_speaker

def process_results(face_to_speaker, transcript):
    results = {
        "face_to_speaker": face_to_speaker,
        "transcript": []
    }
    
    if 'results' in transcript:
        if 'speaker_labels' in transcript['results']:
            segments = transcript['results']['speaker_labels']['segments']
            items = transcript['results'].get('items', [])
            
            for segment in segments:
                speaker = segment['speaker_label']
                face_id = next((face_id for face_id, data in face_to_speaker.items() if data['speaker_label'] == speaker), None)
                
                segment_items = [
                    item for item in items 
                    if 'start_time' in item and 'end_time' in item
                    and float(item['start_time']) >= float(segment['start_time']) 
                    and float(item['end_time']) <= float(segment['end_time'])
                ]
                
                content = ' '.join([item['alternatives'][0]['content'] for item in segment_items if 'alternatives' in item])
                
                results['transcript'].append({
                    "start_time": segment['start_time'],
                    "end_time": segment['end_time'],
                    "speaker": speaker,
                    "face_id": face_id,
                    "content": content
                })
        else:
            items = transcript['results'].get('items', [])
            content = ' '.join([item['alternatives'][0]['content'] for item in items if 'alternatives' in item])
            results['transcript'].append({
                "start_time": items[0]['start_time'] if items and 'start_time' in items[0] else None,
                "end_time": items[-1]['end_time'] if items and 'end_time' in items[-1] else None,
                "speaker": "unknown",
                "face_id": None,
                "content": content
            })
    else:
        print("Unexpected transcript format")
    
    return results

if __name__ == "__main__":
    video_path = './temp/xyz.mp4'
    bucket_name = 'replace with your bucket name'
    s3_key = 'Senators.mp4'
    process_video(video_path, bucket_name, s3_key)

