## 기본 정보 확인
import os
import glob

folder_path = "/Users/garden/Desktop/AWS_AIML/data/custom-dataset/train/"
file_list = os.listdir(folder_path)

jpg_list = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
png_list = [file for file in os.listdir(folder_path) if file.endswith('.png')]
all_image = jpg_list + png_list
print('이미지 총 개수 :', len(all_image))
# 이미지 총 개수 : 26271

# 파일명에 _aug 포함 : 바운딩박스 정보 존재 이미지
jpg_aug_list = [file for file in glob.glob(folder_path + f'/*{"_aug"}*.jpg')]
png_aug_list = [file for file in glob.glob(folder_path + f'/*{"_aug"}*.png')]
print('가공 이미지 :', len(jpg_aug_list)+len(png_aug_list))

# 파일명에 _aug 미포함 : 원본 이미지
raw_list = [file for file in all_image if '_aug' not in file]
print('원본 이미지 :', len(raw_list))

lr = [file for file in raw_list if '_lr' in file]
print('lr :', len(lr))
ud = [file for file in raw_list if '_ud' in file]
print('ud :', len(ud))
origin = [file for file in raw_list if '_ud' not in file and '_lr' not in file]
print('origin :', len(origin))
# lr 417개 / ud 417개 / 원본 417개 >> 원본을 뒤집거나 좌우반전 한 사진


## YOLO dataset 구조에서 class 개수 세기
import datetime
import time
import copy

name_dict = {}
read_count = 0

start_time = time.time()

txt_file = glob.glob(os.path.join(folder_path, "*.txt"))
label_path_list = [file for file in txt_file if '_ud' not in file and '_lr' not in file and '_aug' not in file]

read_count = 0
class_count_dict = {}
    
for label_path in label_path_list:
    with open(label_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            line_split = line.split(" ")
            if line_split[0] in class_count_dict:
                class_count_dict[line_split[0]] += 1
            else:
                class_count_dict[line_split[0]] = 1
    read_count += 1
    
    if read_count % 1000 == 0:
        print(f"{read_count} / {len(label_path_list)}")
        
name_dict['train'] = copy.deepcopy(class_count_dict)
 
print('\n< Result >')
 
# directory의 class 별 object 개수
for name in name_dict.keys():
    print(f"[{name}]")
    key_list = list(name_dict[name].keys())
    key_list.sort(key=lambda x: int(x))

    print(key_list)
    for key in key_list:
        print("class:", key, "count:", name_dict[name][key])
    print('')
    
print(f"Total Time: {datetime.timedelta(seconds=int(time.time() - start_time))}")
