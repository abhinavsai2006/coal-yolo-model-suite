import os
import shutil
def create_dirs():
    for split in ['train', 'val']:
        os.makedirs(f'dataset/images/{split}', exist_ok=True)
        os.makedirs(f'dataset/labels/{split}', exist_ok=True)

def move_images_and_labels(data_dir, split='train', ratio=0.8):
    class_names = [
        'destructive_coal',
        'fully_pulverized_coal',
        'non_destructive_coal',
        'not_coal',
        'pulverized_coal',
        'strongly_destructive_coal'
    ]
    for idx, cname in enumerate(class_names):
        folder = os.path.join(data_dir, cname)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        n_train = int(len(files) * ratio)
        for i, fname in enumerate(files):
            src_img = os.path.join(folder, fname)
            label_name = os.path.splitext(fname)[0] + '.txt'
            src_label = os.path.join(folder, label_name)
            if not os.path.exists(src_label):
                continue  # skip if no label
            split_dir = 'train' if i < n_train else 'val'
            dst_img = f'dataset/images/{split_dir}/{cname}_{fname}'
            dst_label = f'dataset/labels/{split_dir}/{cname}_{label_name}'
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

def create_yaml():
    yaml_content = '''train: dataset/images/train
val: dataset/images/val
nc: 6
names: ["destructive_coal", "fully_pulverized_coal", "non_destructive_coal", "not_coal", "pulverized_coal", "strongly_destructive_coal"]
'''
    with open('coal.yaml', 'w') as f:
        f.write(yaml_content)

def main():
    create_dirs()
    move_images_and_labels('data')
    create_yaml()
    print('Dataset organized and coal.yaml created.')

if __name__ == '__main__':
    main()
