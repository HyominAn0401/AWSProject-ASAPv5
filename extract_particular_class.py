## 0, 3, 5, 18, 23, 27, 34, 36, 39, 40 클래스 포함한 데이터만 추출
import os
import glob
import copy
import shutil

folder_path = "/Users/garden/Desktop/AWS_AIML/data/custom-dataset/train/"
target_cls = [0, 3, 5, 18, 23, 27, 34, 36, 39, 40]
txt_list = glob.glob(os.path.join(folder_path, "*.txt"))
os.mkdir('./data/custom-dataset/new_train')

read_count = 0
lst = []
name_dict = {}
class_count_dict = {}
    
for label_path in txt_list:
    with open(label_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        temp = []
        for line in lines:
            line_split = line.split(" ")
            temp.append(int(float(line_split[0])))

        if any(cls in target_cls for cls in temp):
            lst.append(label_path)
            shutil.move(label_path,'data/custom-dataset/new_train')
            try:
                shutil.move(label_path[:-4]+'.png','data/custom-dataset/new_train')
            except:
                shutil.move(label_path[:-4]+'.jpg','data/custom-dataset/new_train')

        else:
            continue

        for line in lines:
            line_split = line.split(" ")
            line_split[0] = int(float(line_split[0]))
            if line_split[0] in class_count_dict:
                class_count_dict[line_split[0]] += 1
            else:
                class_count_dict[line_split[0]] = 1

    read_count += 1
    
    if read_count % 1000 == 0:
        print(f"{read_count} / {len(txt_list)}")
        
name_dict['train'] = copy.deepcopy(class_count_dict)
 
print('\n< Result >')
 
# directory의 class 별 object 개수
for name in name_dict.keys():
    print(f"[{name}]")
    key_list = list(name_dict[name].keys())
    key_list.sort(key=lambda x: int(float(x)))

    print(key_list)
    for key in key_list:
        print("class:", key, "count:", name_dict[name][key])
    print('')

print('최종 Train 파일 개수 :', len(lst))
