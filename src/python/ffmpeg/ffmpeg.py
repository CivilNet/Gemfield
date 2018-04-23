import os
import sys
import subprocess
import csv
import random

mp4_suffix_list = ['.mp4','.MP4','.Mp4']
csv_suffix_list = ['.7th.csv']
cat_list = ['background', 'Unlike', 'Shot', 'Preshot', 'Head', 'DFKick', 'Penalty', 'Corner', 'Replay']

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
        #get rid of these videos.
        if 'corner_' in input_video_file:
            continue
        if 'Header' in input_video_file:
            continue
        if 'penalty' in input_video_file:
            continue
        if 'DFKick' in input_video_file:
            continue

        print('get random clip from {}'.format(input_video_file))
        input_video_file_base = input_video_file[:-4]

        seconds_start = calSeconds(start_time)
        if seconds_start == 0:
            continue
        random_start = None
        for i in range(100):
            #print('gemfield: {}'.format(seconds_start))
            random_start_tmp = random.randrange(0, seconds_start)
            if random_start_tmp in highlight_mask[input_video]:
                #print('start frame collision, pick another one...')
                continue
            random_start = random_start_tmp
            break

        if random_start is None:
            print('Cannot find negative samples in this video {}'.format(input_video))
            continue

        new_clip_list.append( (input_video, 'background', str(random_start), str(3) )  )
    return new_clip_list
        
def extractVideoFromCsv(csv_file, video_file):
    clip_list = []
    print('parsing {}'.format(csv_file))
    with open(csv_file) as csvfile:
        data = csvfile.read().decode("utf-8-sig").encode("utf-8")
        readCSV = []
        lines = data.split('\n')
        for l in lines:
            l = l.strip()
            if l == '':
                continue
            readCSV.append(l.split(','))
        #readCSV = csv.reader(csvfile, delimiter=',')

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
            cat_items = cat.split(';')
            for cat_item in cat_items:
                if cat_item not in cat_list:
                    raise Exception('illegal category in {}: {}'.format(video_file, cat_item))

            if len(cat_items) > 1:
                for cat_item in cat_items:
                    #with priority
                    if cat_item == 'Unlike':
                        cat = 'Unlike'
                        break

                    if cat_item == 'Head':
                        cat = 'Head'
                        break

                    if cat_item == 'DFKick':
                        continue
                    cat = cat_item
                print('Finally pickup {} as our category...'.format(cat))
            #do a time check
            try:
                seconds_start = calSeconds(start_time)
                seconds_end = calSeconds(end_time)
                duration = seconds_end - seconds_start
            except Exception as e:
                print('illegal time format: {} and {} in {}. More info:{}'.format(start_time, end_time, csv_file, str(e)))
                sys.exit(1)

            # if duration > 10:
            #     raise Exception('Video clip too long! {} : {} -> {}'.format(video_file, start_time, end_time))
            if cat == 'Replay':
                if duration > 60:
                    raise Exception('Video clip duration should be less than 60! {} : {} -> {}'.format(video_file, start_time, end_time))
            elif duration != 3:
                raise Exception('Video clip duration should be 3! {} : {} -> {}'.format(video_file, start_time, end_time))
            
            clip_list.append( (video_file, cat, start_time, str(duration)) )

            if not video_file in highlight_mask:
                highlight_mask[video_file] = set()

            for frame_number in range(seconds_start - 4, seconds_start + duration + 10):
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

        mp4_suffix = csv_file[-4:]

        suffix = csv_file[-8:]
        if suffix not in csv_suffix_list:
            continue

        video_file_base = csv_file[:-8]
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
    #background_clip_list = genRandomClip(clip_list)

    #if len(background_clip_list) > 420:
    #    random.shuffle(background_clip_list)
    #    background_clip_list = background_clip_list[:420]

    with open('background.txt','w') as f_bg, open('positive_samples.txt', 'w') as f_pos:
        #for i in background_clip_list:
        #    f_bg.writelines('{},{},{},{}\n'.format(i[0], i[1], i[2], i[3] ))
        for i in clip_list:
            f_pos.writelines('{},{},{},{}\n'.format(i[0], i[1], i[2], i[3] ))
            
    doExtract(clip_list, True)
    #doExtract(background_clip_list, True)



    




