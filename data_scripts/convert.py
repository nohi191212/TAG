import json
import os
import matplotlib
import threading
matplotlib.use('Agg')  # Set backend to prevent displaying figures
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from queue import Queue
# 将VCR数据转换为A-OKVQA格式
def read_anno_from_metadata(metadata_file):
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    boxes = data["boxes"]
    segs = data["segms"]
    names = data["names"]
    return boxes, segs, names


def make_sentence(list_of_words, object_names):
    sentence = ''
    for w in list_of_words:
        if w in [',' , '.', '!', '?', ';', ':', '\'']:
            sentence += w # 标点符号前不需要加空格
        elif isinstance(w, list):
            # 数字的情况
            if len(w) == 1:
                if sentence == '':
                    number_part = object_names[int(w[0])]
                else:
                    number_part = ' '+object_names[int(w[0])]
            else:
                number_part = ' and'.join([object_names[int(x)] for x in w])
            sentence += number_part
        else:
            if sentence == '':
                sentence += w
            else:
                if sentence[-2:] == 'n\'':
                    sentence += w
                else:
                    sentence += ' '+w # 加空格 
    return sentence

def draw_annotations(image, annotations):
    # Load image and convert to RGBA
    draw = ImageDraw.Draw(image)
    
    linewidth = int(0.005 * max(image.size))
    font_size = int(0.025 * max(image.size))
    
    # Prepare color map
    num_objects = len(annotations)
    colors = plt.cm.get_cmap('tab20c', len(annotations))
    
    # Track class counts and color assignment
    class_counts = {}
    object_colors = {}

    for idx, ann in enumerate(annotations):
        class_name = ann['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        new_name = f"{class_name}{class_counts[class_name]}"
        
        # Assign unique color
        color = colors(idx)
        r, g, b = color[:3]
        
        # Define RGBA colors with alpha
        bbox_color = (int(r*255), int(g*255), int(b*255), 128)  # Alpha: 0.5
        text_color = (int(r*255), int(g*255), int(b*255), 128)  # Alpha: 0.5
        seg_color = (int(r*255), int(g*255), int(b*255), 80)    # Alpha: 0.3

        # Draw bounding box
        x1, y1, x2, y2, _ = ann['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=linewidth)
        
        # Draw segmentation polygon
        if len(ann['segmentation']) > 0:
            segment = ann['segmentation'][0]
            for i in range(len(segment)):
                p1 = segment[i]
                p2 = segment[(i+1) % len(segment)]
                draw.line([p1[0], p1[1], p2[0], p2[1]], fill=seg_color, width=int(linewidth*0.7))
        
        # Draw text box
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        left, top, right, bottom = font.getbbox(new_name)
        text_width = right - left
        text_height = bottom - top
        
        padding = 5
        textbox_left = x1 
        textbox_bottom = y1 - text_height - padding 
        textbox_right = x1 + text_width + padding * 2
        textbox_top = y1 + padding
        
        # Adjust textbox position if out of image bounds
        if (textbox_bottom + textbox_top) / 2 < 0:
            textbox_left = x1 
            textbox_bottom = y2 - padding
            textbox_right = x1 + text_width + padding * 2
            textbox_top = y2 + text_height + padding 
        if (textbox_bottom + textbox_top) / 2 > image.size[1]:
            textbox_left = x1 
            textbox_bottom = y1
            textbox_right = x1 + text_width + padding * 2
            textbox_top = y1 + text_height + padding * 2
        
        textbox = [textbox_left, textbox_bottom, textbox_right, textbox_top]
        draw.rectangle(textbox, fill=text_color)
        draw.text((x1, textbox_bottom), new_name, font=font, fill=(0, 0, 0, 255))
        
    return image


def process_item(item, vcr_dir, split, result_queue):
    # Extract relevant fields from VCR data
    movie = item["movie"]
    objects = item["objects"] # a list of object names , for example
    # ["person","person","person", "car", "cellphone", "clock"],
    new_objects = []
    class_counts_ = {}
    for obj in objects:
        class_counts_[obj] = class_counts_.get(obj, 0) + 1
        new_objects.append(f"{obj}{class_counts_[obj]}")
    
    img_fn = item["img_fn"]
    img_file = os.path.join(vcr_dir, 'vcr1images', img_fn)
    
    metadata_fn = item["metadata_fn"]
    metadata_file = os.path.join(vcr_dir, 'vcr1images', metadata_fn)
    
    # question_orig = item["question_orig"]
    rationale_orig = item["rationale_orig"]
    question = item["question"]
    question = make_sentence(question, new_objects)
    answer_choices = item["answer_choices"]
    choices = []
    for choice in answer_choices:
        # 这里是一堆list, list里是单个单词、列表括起来的数字、标点符号.
        sentence = make_sentence(choice, new_objects)
        choices.append(sentence)
        
    answer_label = item["answer_label"]
    question_number = item["question_number"] # int 0, 1, 2, ...
    img_id = item["img_id"] # str "val-0"
        
    
    img_save_path = os.path.join(vcr_dir, f'vcr_{split}', f'{img_id}.jpg')
    if not os.path.exists(img_save_path):
        img_data = Image.open(img_file).convert('RGBA')
        boxes, segs, names = read_anno_from_metadata(metadata_file)
        # assert all(a==b for a,b in zip(objects, names))
        annotations = []
        for box, seg, name in zip(boxes, segs, names):
            annotations.append({
                'class_name': name,
                'bbox': box,
                'segmentation': seg,
            })
        img_data = draw_annotations(img_data, annotations)
        img_data = img_data.convert('RGB')
        img_data.save(img_save_path)
    
    # Create A-OKVQA format entry
    aokvqa_entry = {
        "split": "val",  # Assuming validation split; adjust as needed
        "image_id": question_number, # should be int
        "question_id": img_id,  # the basename of the image file
        "question": question,
        "choices": choices,  # Extract the first part of each answer choice
        "correct_choice_idx": answer_label,
        "direct_answers": [make_sentence(answer_choices[answer_label], new_objects)],  # Use the correct answer choice
        "difficult_direct_answer": False,  # Assuming not difficult; adjust as needed
        "rationales": [rationale_orig],  # Include all rationale choices
    }    
    result_queue.put(aokvqa_entry)


from tqdm import tqdm
def convert_vcr_to_aokvqa(vcr_data, vcr_dir, split):
    result_queue = Queue()
    threads = []
    thread_count = 0
    max_threads = 100
    
    for item in tqdm(vcr_data):
        while thread_count >= max_threads:
            # 等待线程数量小于 max_threads
            for thread in threads:
                if not thread.is_alive():
                    threads.remove(thread)
                    thread_count -= 1
                    break

        thread = threading.Thread(target=process_item, args=(item, vcr_dir, split, result_queue))
        thread.start()
        threads.append(thread)
        thread_count += 1
        
        
    for thread in threads:
        thread.join()
        
        
    aokvqa_data = []
    while not result_queue.empty():
        aokvqa_data.append(result_queue.get())
    return aokvqa_data

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcr_dir", type=str, default="/home1/caisihang/project/TAG/datasets/vcr", dest="vcr_dir")
    parser.add_argument("--split", type=str, default="val", help="Split to convert (train/val/test)", dest="split")
    args = parser.parse_args()
    vcr_dir = args.vcr_dir
    split = args.split
    
    jsonl_file = os.path.join(vcr_dir, f"{split}.jsonl")
    
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data
    # Load VCR data from a JSON file
    vcr_data = read_jsonl(jsonl_file)
    # with open("/home1/caisihang/project/TAG/datasets/vcr/val_clean.json", "w") as f:
    #     json.dump(vcr_data, f, indent=4)
    # quit()
    # Convert VCR data to A-OKVQA format
    aokvqa_data = convert_vcr_to_aokvqa(vcr_data, vcr_dir=vcr_dir, split=split)

    # Save the converted data to a new JSON file
    with open(f"/home1/caisihang/project/TAG/datasets/vcr/{split}_aokvqa_format.json", "w") as f:
        json.dump(aokvqa_data, f, indent=4)