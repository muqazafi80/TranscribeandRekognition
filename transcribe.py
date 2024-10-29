import boto3
import time
import json
import urllib.request

# Set up the AWS clients
transcribe = boto3.client('transcribe')
s3 = boto3.client('s3')

# S3 bucket and file details
bucket_name = 'your buckeet name'
s3_key = 'sample.mp4'

# Start the transcription job
job_name = f"transcription-job-{int(time.time())}"
job_uri = f"s3://{bucket_name}/{s3_key}"

transcribe.start_transcription_job(
    TranscriptionJobName=job_name,
    Media={'MediaFileUri': job_uri},
    MediaFormat='mp4',
    LanguageCode='en-US',
    Settings={'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 10}
)

# Wait for the transcription job to complete
while True:
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(5)

if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
    # Get the transcript
    transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    
    # Download the transcript file
    with urllib.request.urlopen(transcript_uri) as response:
        transcript = json.loads(response.read().decode('utf-8'))

    # Process and print the transcript
    segments = transcript['results']['speaker_labels']['segments']
    items = transcript['results']['items']

    current_speaker = None
    current_start = None
    current_text = []

    for item in items:
        if 'start_time' in item:
            start_time = float(item['start_time'])
            end_time = float(item['end_time'])

            # Find the corresponding speaker
            for segment in segments:
                if start_time >= float(segment['start_time']) and end_time <= float(segment['end_time']):
                    speaker = f"Speaker {segment['speaker_label']}"
                    break

            if speaker != current_speaker:
                if current_speaker:
                    print(f"{current_speaker} ({current_start:.2f}s - {end_time:.2f}s): {' '.join(current_text)}")
                current_speaker = speaker
                current_start = start_time
                current_text = []

            current_text.append(item['alternatives'][0]['content'])

        elif item['type'] == 'punctuation':
            current_text[-1] += item['alternatives'][0]['content']

    # Print the last speaker's text
    if current_speaker:
        print(f"{current_speaker} ({current_start:.2f}s - {end_time:.2f}s): {' '.join(current_text)}")

else:
    print("Transcription job failed.")
