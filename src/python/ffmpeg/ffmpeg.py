import os
import sys
import subprocess
import csv
import random

mp4_suffix_list = ['.mp4','.MP4','.Mp4']
csv_suffix_list = ['.csv','.CSV','.Csv']

highlight_mask = {}

def doExtract(clip_list, is_image=False):
    for clip in clip_list:
        extractVideo(clip, is_image)

def extractVideo(clip, is_image=False):
    input_video, cat, start_time, duration = clip
    input_video_file = os.path.basename(input_video)
    input_video_file_base = input_video_file[:-4]

    images_folder = '{}__{}__{}__{}'.format(cat, input_video_file_base, start_time.replace(':','-'), duration)

    if not os.path.isdir(images_folder):
        os.makedirs(images_folder)

    if is_image:
        output_images = '{}/%6d.jpg'.format(images_folder)
        ffmpeg_cmd = "ffmpeg -ss {} -i {} -t {} {}".format(start_time, input_video, duration, output_images)
    else:
        output_video = '{}/{}__{}__{}__{}.mp4'.format(input_video_file_base, cat, input_video_file_base, start_time.replace(':','-'), duration)
        ffmpeg_cmd = "ffmpeg -i {} -vcodec copy -acodec copy -ss {} -t {} {}".format(input_video, start_time, duration, output_video)
    print('will execute {}'.format(ffmpeg_cmd))

    return_code = subprocess.call(ffmpeg_cmd, shell=True)

    if return_code != 0:
        print("[Error] {} failed".format(ffmpeg_cmd))
        sys.exit(1)

def calSeconds(time_str):
    if ' ' in time_str:
        raise Exception('whitespace in time string is not allowed.')
    h,m,s = time_str.split(':')
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    return seconds


def genRandomClip(clip_list):
    new_clip_list = []
    for clip in clip_list:
        input_video, cat, start_time, duration = clip
        input_video_file = os.path.basename(input_video)
        input_video_file_base = input_video_file[:-4]

        seconds_start = calSeconds(start_time)
        random_start = None
        for i in range(100):
            random_start_tmp = random.randrange(0, seconds_start)
            if random_start_tmp in highlight_mask[input_video]:
                print('start frame collision, pick another one...')
                continue
            random_start = random_start_tmp
            break

        if random_start is None:
            raise Exception('Cannot find negative samples in this video {}'.format(input_video))

        new_clip_list.append( (input_video, 'background', str(random_start), str(6) )  )
    return new_clip_list
        


def extractVideoFromCsv(csv_file, video_file):
    clip_list = []
    print('parsing {}'.format(csv_file))
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if len(row) == 0:
                continue
            if len(row) < 3:
                raise Exception('illegal row: {}'.format(str(row)))
            print('category: {} | start time: {} | end time: {}'.format(row[0],row[1],row[2]))
            cat = row[0].strip()
            start_time = row[1].strip()
            end_time = row[2].strip()
            if cat == '' and start_time == '' and end_time == '':
                print('[Warning] found empty row: {}'.format(row))
                continue
            #do a time check
            try:
                seconds_start = calSeconds(start_time)
                seconds_end = calSeconds(end_time)
                duration = seconds_end - seconds_start
            except Exception as e:
                print('illegal time format: {} and {} in {}. More info:{}'.format(start_time, end_time, csv_file, str(e)))
                sys.exit(1)

            if duration > 10:
                raise Exception('Video clip too long! {} : {} -> {}'.format(video_file, start_time, end_time))

            if duration < 4:
                raise Exception('Video clip too short! {} : {} -> {}'.format(video_file, start_time, end_time))
            
            clip_list.append( (video_file, cat, start_time, str(duration)) )

            if not video_file in highlight_mask:
                highlight_mask[video_file] = set()

            for frame_number in range(seconds_start - 7, seconds_start + duration + 1):
                highlight_mask[video_file].add(frame_number)
            

    return clip_list


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <csv_dir>'.format(sys.argv[0]))
        sys.exit(1)

    clip_list = []

    csv_dir = sys.argv[1]
    for csv_file in os.listdir(csv_dir):
        if csv_file.startswith('.'):
            print('found hidden file: {}'.format(csv_file))
            continue

        suffix = csv_file[-4:]
        if suffix not in mp4_suffix_list and suffix not in csv_suffix_list:
            raise Exception('illegal file: {}'.format(csv_file))

        if suffix not in csv_suffix_list:
            continue
        video_file_base = csv_file[:-4]
        csv_file = os.path.join(csv_dir, csv_file)
        video_file = None
        for suffix in ['.MP4','.mp4','.Mp4']:
            video_file_tmp = '{}{}'.format(video_file_base, suffix)
            video_file_tmp = os.path.join(csv_dir, video_file_tmp)
            if os.path.isfile(video_file_tmp):
                video_file = video_file_tmp
                break
            
        if video_file is None:
            raise Exception('The corresponding mp4 of {} not found.'.format(csv_file))

        clip_list_tmp = extractVideoFromCsv(csv_file, video_file)
        clip_list.extend(clip_list_tmp)

    #doExtract(clip_list, True)
    background_clip_list = genRandomClip(clip_list)
    for i in background_clip_list:
        print(i)

    
    doExtract(background_clip_list, True)



    




