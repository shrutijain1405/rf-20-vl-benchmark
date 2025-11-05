import os
import json
import glob
from tqdm import tqdm
import concurrent.futures
import threading
import logging
import argparse
import pickle
import copy
from qwen_vl_utils import smart_resize #expects qwen-vl-utils==0.0.8
from utils.qwen_eval_utils import *
from utils.shared_eval_utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch


NUM_FEW_SHOT_EXAMPLES = 3
# MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
# MODEL_NAME = "Qwen3-VL-4B-Instruct"
# MODEL_NAME = "Qwen3-VL-8B-Instruct"
# MODEL_NAME = "Qwen2.5-VL-72B-Instruct"
# MODEL_NAME = "Qwen3-VL-235B-A22B-Instruct-FP8"

MODEL_DESC = ""
# ROOT_DIR = "/data3/spjain/rf20-vl-fsod"
DATASET = {
    0 : ["actions", "aerial-airport"],
    1 : ["aquarium-combined", "defect-detection", "dentalai", "trail-camera"],
    3 : ["gwhd2021", "lacrosse-object-detection", "all-elements"],
    4 : ["soda-bottles", "orionproducts", "wildfire-smoke"],
    5 : ["paper-parts"],
    6 : ["new-defects-in-wood", "the-dreidel-project","recode-waste"],
    7 : ["flir-camera-objects", "water-meter", "wb-prova","x-ray-id"]
}


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rf100_qwen_multi_class_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=100):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qwen_model(model_name):
    
   
    model = None
    if(model_name.startswith("Qwen2.5-VL")):
         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/"+model_name,
        torch_dtype= torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    elif(model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
        model = LLM(
            model="Qwen/"+model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.80,
            enforce_eager=False,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0
        )
        
    elif(model_name.startswith("Qwen3-VL-235B") or model_name.startswith("Qwen3-VL-30B")):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/"+model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        ) 
    elif(model_name.startswith("Qwen3-VL")):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/"+model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
    else:
        print("Error: Invalid model name")
        return None, None

    processor = AutoProcessor.from_pretrained("Qwen/"+model_name)
    model.eval()

    print("processor.tokenizer.padding_side:", processor.tokenizer.padding_side)
    print("processor.tokenizer.pad_token:", processor.tokenizer.pad_token)
    print("processor.tokenizer.eos_token:", processor.tokenizer.eos_token)

    # ✅ Fix padding side and pad token
    tokenizer = processor.tokenizer  # access the underlying tokenizer

    tokenizer.padding_side = "left"  # left padding for decoder-only models


    # 2. Keep "<|endoftext|>" as pad token (already set)
    # Do NOT overwrite with eos_token ("<|im_end|>")

    # # Ensure PAD token exists
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # Save these updates into the processor so it uses them
    processor.tokenizer = tokenizer

    print("processor.tokenizer.padding_side:", processor.tokenizer.padding_side)
    print("processor.tokenizer.pad_token:", processor.tokenizer.pad_token)
    print("processor.tokenizer.eos_token:", processor.tokenizer.eos_token)


    # Optionally, if you want to persist this behavior for future loads:
    # processor.save_pretrained("./qwen2_5_vl_leftpad")

    return model, processor

def inference(messages, model, processor, model_name):
    set_seed()
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # generated_ids = model.generate(**inputs, max_new_tokens=1024)
        if(model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=2048,
                top_k=-1,
                stop_token_ids=[],
            )
            generated_ids = model.generate(**inputs, sampling_params)
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=2048)


    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    #Sample
    # ```json
    #     [
    #         {"bbox_2d": [1145, 563, 1427, 920], "label": "Adult", "score": 0.95}
    #     ]
    # ```
    
    return output_text
            

def format_ground_truth_for_model_qwen(annotations_for_image, categories_dict, width, height):
    """
    Formats ground truth annotations for a single image into the
    JSON string expected in the 'assistant' turn of a few-shot prompt for Qwen.
    Note: Qwen expects pixel coordinates, not normalized.
    """
    output_boxes = []
    for ann in annotations_for_image:
        cat_id = ann['category_id']
        cat_name = categories_dict.get(cat_id, "unknown")

        x, y, w, h = ann['bbox']
        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        output_boxes.append({
            "label": cat_name,
            "bbox_2d": [xmin, ymin, xmax, ymax]
        })

    json_string = json.dumps(output_boxes, indent=2)
    return json_string

def create_few_shot_messages_qwen(train_folder, categories_dict, model_name):
    """
    Creates few-shot examples as a list of alternating user/assistant message dicts
    for the Qwen API.

    Args:
        train_folder (str): Path to the training data folder.
        categories_dict (dict): Mapping from category ID to category name.

    Returns:
        list[dict]: A list of message dictionaries for the few-shot sequence.
                    Returns empty list if annotations or images are insufficient.
    """
    annotation_file = os.path.join(train_folder, "_annotations.coco.json")

    with open(annotation_file, 'r') as f:
        train_data = json.load(f)

    images_by_id = {img['id']: img for img in train_data.get('images', [])}

    annotations_by_image = {}
    for ann in train_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    few_shot_messages = []
    annotated_image_ids = list(annotations_by_image.keys())

    category_prompt = ", ".join(categories_dict.values())
    user_prompt_text_for_example = f"Locate all of the following objects: {category_prompt} (each of those is a separate class) in the image and output the coordinates in JSON format."

    examples_added = 0
    for img_id in annotated_image_ids:
        if(examples_added >= NUM_FEW_SHOT_EXAMPLES):
            break
        img_info = images_by_id[img_id]
        original_width = img_info['width']
        original_height = img_info['height']
        input_height = 1000
        input_width = 1000
        if(model_name.startswith("Qwen2.5-VL")):
            input_height, input_width = smart_resize(original_height, original_width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        image_path = os.path.join(train_folder, img_info['file_name'])

        base64_image_example = encode_image(image_path)

        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "image_url": f"data:image/jpeg;base64,{base64_image_example}"
                },
                {"type": "text", "text": user_prompt_text_for_example},
            ]
        }

        example_annotations = annotations_by_image[img_id]

        scaled_annotations = []
        for ann in example_annotations:
            x, y, w, h = ann['bbox']
            x_scaled = x * input_width / original_width
            y_scaled = y * input_height / original_height
            w_scaled = w * input_width / original_width
            h_scaled = h * input_height / original_height

            scaled_ann = ann.copy()
            scaled_ann['bbox'] = [x_scaled, y_scaled, w_scaled, h_scaled]
            scaled_annotations.append(scaled_ann)

        assistant_response_text = format_ground_truth_for_model_qwen(
            scaled_annotations, categories_dict, input_width, input_height
        )

        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response_text}]
        }

        few_shot_messages.append(user_message)
        few_shot_messages.append(assistant_message)
        examples_added += 1

    logger.info(f"Successfully created {examples_added} few-shot example turns (x2 message dicts).")
    return few_shot_messages

def precompute_prompts_for_dataset_qwen(dataset_dir, categories, categories_dict, model_name):
    """
    Precompute prompts for Qwen, including multi-turn few-shot.
    MODIFIED: Passes instructions separately for 'combined' mode to mimic Gemini logic.
    """
    dataset_name = os.path.basename(dataset_dir)
    train_folder = os.path.join(dataset_dir, "train")

    readme_path = os.path.join(dataset_dir, "README.dataset.txt")
    instructions = ""
    with open(readme_path, 'r') as f:
        instructions = f.read().strip()
        logger.info(f"Loaded instructions from {readme_path} for {dataset_name}")

    basic_system_prompt = SYSTEM_PROMPT

    category_prompt = ", ".join(categories)
    final_query_text_basic = f"Locate all of the following objects: {category_prompt} in the image and output the coordinates in JSON format." # Saying (each of those is a separate class) might be helpful, but dropped for now

    instructions_system_prompt = basic_system_prompt
    if instructions:
         final_query_text_instructions_standalone = f"Locate all of the following objects: {category_prompt} in the image and output the coordinates in JSON format.\n\nUse the following annotator instructions to improve detection accuracy:\n{instructions}\n"
    else:
         final_query_text_instructions_standalone = final_query_text_basic

    few_shot_messages_list = []
    few_shot_messages_list = create_few_shot_messages_qwen(train_folder, categories_dict, model_name)

    few_shot_system_prompt = basic_system_prompt
    final_query_text_few_shot = final_query_text_basic

    combined_system_prompt = basic_system_prompt
    final_query_text_combined = final_query_text_basic

    prompts = {
        'basic': (basic_system_prompt, None, final_query_text_basic, None),
        'instructions': (instructions_system_prompt, None, final_query_text_instructions_standalone, instructions),
        'few_shot': (few_shot_system_prompt, few_shot_messages_list, final_query_text_few_shot, None),
        'combined': (combined_system_prompt, few_shot_messages_list, final_query_text_combined, instructions)
    }

    return prompts

def convert_to_coco_format(qwen_predictions, image_id, original_width, original_height, 
                            input_width, input_height, categories_dict):
    """Convert Qwen detection results to COCO format"""
    coco_annotations = []
    
    boxes = parse_qwen_response(qwen_predictions, logger)
    
    for idx, box in enumerate(boxes):
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("bbox_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label = box.get("label", "unknown")
        
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label.lower():
                category_id = cat_id
                break
        
        if category_id is None:
            continue
            
        x1, y1, x2, y2 = bbox_2d
        x1 = int(x1 / input_width * original_width)
        y1 = int(y1 / input_height * original_height)
        x2 = int(x2 / input_width * original_width)
        y2 = int(y2 / input_height * original_height)
        
        width = x2 - x1
        height = y2 - y1
        
        coco_annotation = {
            "id": idx,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "segmentation": [],
            "iscrowd": 0,
            "score": 1.0
        }
        
        coco_annotations.append(coco_annotation)
    
    return coco_annotations, boxes

def process_image(args):
    """
    Process a single image using the determined prompt strategy.
    MODIFIED: In 'combined' mode, prepends instructions to the text of the
    first user message in the few-shot sequence, mimicking Gemini.
    """
    set_seed()

    (image_info, image_id, file_name, height, width,
     test_folder, categories_dict, results_dir, vis_dir,
     prompts, few_shot, just_instructions, combined, status_file,
     status_file_lock, model, processor, model_name) = args

    

    status_dict = {}
    image_key = str(image_id)

    with status_file_lock:
        try:
            if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
                with open(status_file, "rb") as sf:
                    status_dict = pickle.load(sf)
            if not isinstance(status_dict, dict):
                 logger.warning(f"Status file {status_file} contained non-dict data. Resetting.")
                 status_dict = {}
        except (EOFError, pickle.UnpicklingError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Error reading status file {status_file} ({e}). Treating as empty.")
            status_dict = {}
    
    if status_dict.get(image_key) is True:
         logger.info(f"Skipping image {file_name} (ID: {image_id}) - Already marked True in status.pkl.")
         print(f"Skipping image {file_name} (ID: {image_id}) - Already marked True in status.pkl.")
         return None, None, True, image_key
    
    image_path = os.path.join(test_folder, file_name)
    if not os.path.exists(image_path):
        image_files = glob.glob(os.path.join(test_folder, "**", file_name), recursive=True)
        image_path = image_files[0]

    current_attempt_status = False
    coco_annotations = []
    error_message_for_return = None
    original_boxes_for_vis = []

    try:
        base64_image_test = encode_image(image_path)

        mode = 'basic'
        if combined: mode = 'combined'
        elif few_shot: mode = 'few_shot'
        elif just_instructions: mode = 'instructions'

        system_prompt, few_shot_message_list_orig, final_query_text, instructions_text = prompts[mode]

        final_messages = []

        if system_prompt:
             final_messages.append({
                 "role": "system",
                 "content": [{"type": "text", "text": system_prompt}]
             })

        processed_few_shot_messages = []
        if few_shot_message_list_orig:
            few_shot_messages = copy.deepcopy(few_shot_message_list_orig)

            if mode == 'combined' and instructions_text:
                if few_shot_messages and few_shot_messages[0].get("role") == "user":
                    first_user_message = few_shot_messages[0]
                    content = first_user_message.get("content", [])
                    text_part_modified = False
                    for i, part in enumerate(content):
                        if part.get("type") == "text":
                            original_text = part.get("text", "")
                            part["text"] = f"{original_text}\n\nUse the following annotator instructions:\n{instructions_text}\n"
                            text_part_modified = True
                            logger.debug(f"Prepended instructions to text part {i} in the first user turn for image {image_id}.")
                            break

            processed_few_shot_messages = few_shot_messages

        final_messages.extend(processed_few_shot_messages)
        
        input_height = 1000
        input_width = 1000
        if(model_name.startswith("Qwen2.5-VL")):
            input_height, input_width = smart_resize(height, width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        # import pdb;pdb.set_trace()
        final_user_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "image_url": f"data:image/jpeg;base64,{base64_image_test}" # Assuming JPEG
                },
                {"type": "text", "text": final_query_text},
            ],
            "temperature": 0.0
        }
        final_messages.append(final_user_message)

        logger.info(f"INITIATING MODEL call for image {file_name} (ID: {image_id}) using mode '{mode}' with {len(final_messages)} total messages.")

        response_text  = inference(
            messages=final_messages,
            model=model,
            processor = processor,
            model_name = model_name
        )

        coco_annotations, original_boxes_for_vis = convert_to_coco_format(
            response_text, image_id, width, height, input_width, input_height,
            categories_dict
        )

        if not coco_annotations:
                logger.info(f"Image {image_id} ({file_name}): API call successful ({mode}), but no valid boxes detected/parsed from response: {response_text[:100]}...")
                original_boxes_for_vis = []

    except Exception as e:
        error_message = f"Unexpected error processing {file_name} (ID: {image_id}): {e}"
        logger.error(error_message, exc_info=True)
        current_attempt_status = False
        error_message_for_return = error_message

    try:
        vis_save_path = os.path.join(vis_dir, file_name)
        visualize_predictions(
            image_path, original_boxes_for_vis if original_boxes_for_vis else [],
            input_width, input_height,
            width, height,
            categories_dict, vis_save_path, logger
        )
    except Exception as vis_e:
        logger.exception(f"Failed to visualize predictions for {file_name} (ID: {image_id}): {vis_e}")

    return coco_annotations, error_message_for_return, current_attempt_status, image_key

def process_dataset(model, processor, dataset_dir, few_shot, just_instructions, combined, results_dir, vis_dir, model_name,debug):
    """Process a single dataset using the selected mode and status file."""
    dataset_name = os.path.basename(dataset_dir)

    status_file = os.path.join(results_dir, dataset_name+"_status.pkl")
    master_status_dict = {}
    status_file_lock = threading.Lock()

    test_folder = os.path.join(dataset_dir, "test")
    train_folder = os.path.join(dataset_dir, "train")

    annotation_file = glob.glob(os.path.join(test_folder, "*_annotations.coco.json"))
    annotation_file = annotation_file[0]

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    categories = annotations.get("categories", [])
    category_names = [cat["name"] for cat in categories]
    categories_dict = {cat["id"]: cat["name"] for cat in categories}

    prompts = precompute_prompts_for_dataset_qwen(dataset_dir, category_names, categories_dict, model_name)
    
    images = annotations.get("images", [])

    results_file = os.path.join(results_dir, f"predictions_"+dataset_name+".json")
    existing_results = load_existing_results(results_file, logger)

    processed_count = 0
    error_count = 0
    skipped_count = 0

    logger.info(f"Processing dataset: {dataset_name} with model {model_name}")
    logger.info(f"Found {len(images)} images in annotations.")
    logger.info(f"Results will be stored in {results_dir}")
    logger.info(f"Status file: {os.path.join(results_dir, dataset_name+'_status.pkl')}")

    status_file_lock = threading.Lock()

    current_master_status_on_disk = {}
    with status_file_lock:
        if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
            try:
                with open(status_file, "rb") as sf:
                    current_master_status_on_disk = pickle.load(sf)
                if not isinstance(current_master_status_on_disk, dict):
                    logger.warning(f"Loaded status file {status_file} contained non-dict data. Resetting for this run's accumulation.")
                    current_master_status_on_disk = {}
            except (EOFError, pickle.UnpicklingError, ValueError, FileNotFoundError) as e:
                logger.warning(f"Error reading status file {status_file} ({e}). Treating as empty for this run's accumulation.")
                current_master_status_on_disk = {}
    master_status_dict_to_update = current_master_status_on_disk.copy()


    results_map = {}
    for ann in existing_results:
        img_id = ann.get('image_id')
        if img_id is not None:
            if img_id not in results_map:
                results_map[img_id] = []
            results_map[img_id].append(ann)

    total_images_to_process = len(images)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     futures = {executor.submit(process_image, args): args[1] for args in args_list}

    #     with tqdm(total=len(futures), desc=f"Processing {dataset_name}", unit="image", leave=False) as pbar:
    for image_info in tqdm(images, desc=f"Processing {dataset_name}", unit="image"):
        # if(image_info["file_name"] != 'IMG_2279_jpeg_jpg.rf.0a869bcdd1d8dcf306906cc6df1fe9d5.jpg'):
        #     continue
        # # if(processed_count >= 10):
        #     break
        if(debug and processed_count >= 3):
            break
        try:
            img_id = image_info["id"]
            coco_result, error_msg, image_was_successful, processed_image_key_str = process_image((
                image_info, image_info["id"], image_info["file_name"],
                image_info["height"], image_info["width"],
                test_folder,
                categories_dict, results_dir, vis_dir,
                prompts, few_shot, just_instructions, combined,status_file,
                status_file_lock,model, processor, model_name
            ))
            processed_count += 1

            if error_msg:
                error_count += 1
                print(f"Image ID {img_id}: Error - {error_msg}")
                results_map.setdefault(img_id, [])
            elif coco_result is None and error_msg is None:
                skipped_count += 1
                results_map.setdefault(img_id, [])
                logger.debug(f"Image ID {img_id}: Confirmed skip based on status (already True).")
                master_status_dict_to_update[processed_image_key_str] = True
            elif coco_result is not None and error_msg is None:
                results_map[img_id] = coco_result
                if image_was_successful:
                        master_status_dict_to_update[processed_image_key_str] = True
                logger.debug(f"Updated results for image {img_id}. Success: {image_was_successful}")

        except Exception as exc:
            processed_count += 1
            error_count += 1
            logger.error(f"Image ID {img_id} generated an exception during future processing: {exc}", exc_info=True)
            print(f"Image ID {img_id}: Worker exception - {exc}")
            results_map.setdefault(img_id, [])

    final_results_list = []
    for img_id, annotations_list in results_map.items():
        if annotations_list:
            unique_anns = []
            seen_ann_tuples = set()
            for ann in annotations_list:
                 bbox_tuple = tuple(ann.get('bbox', []))
                 ann_tuple = (ann.get('category_id'), bbox_tuple)
                 if bbox_tuple and ann_tuple not in seen_ann_tuples:
                     unique_anns.append(ann)
                     seen_ann_tuples.add(ann_tuple)
            final_results_list.extend(unique_anns)

    try:
        with open(results_file, 'w') as f:
            json.dump(final_results_list, f, indent=2)
    except IOError as e:
        raise Exception(f"Failed to write final results file {results_file}: {e}")

    with status_file_lock: # Use the same lock to protect the write operation
        with open(status_file, 'wb') as sf:
            pickle.dump(master_status_dict_to_update, sf)
    logger.info(f"Successfully saved updated status to {status_file}")

    logger.info(f"Saved results to {results_file}")
    logger.info(f"Final results for {dataset_name}: {len(final_results_list)} unique annotations")
    logger.info(f"Total images checked/attempted for {dataset_name}: {processed_count}/{total_images_to_process}")
    logger.info(f"Images skipped due to existing 'True' status: {skipped_count}")
    logger.info(f"Image-level errors/failures reported: {error_count}")

    return results_file, final_results_list, processed_count, error_count, skipped_count

def main():
    """Main function to process all datasets with different modes."""
    parser = argparse.ArgumentParser(description='Process datasets for object detection using Qwen')
    # Add arguments similar to Gemini
    parser.add_argument('--just_instructions', action='store_true',
                        help='Use custom instructions from README')
    parser.add_argument('--few_shot', action='store_true',
                        help='Use few-shot examples from train set')
    parser.add_argument('--combined', action='store_true',
                        help='Use both instructions and few-shot examples (if available)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional custom root directory to save results and visualizations')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='dir where data is there')
    parser.add_argument('--model_name', type=str, default="Qwen3-VL-235B-A22B-Instruct",
                        help='model name')
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id to use")
    parser.add_argument("--parallel", action='store_true', help="Whether to divide datasets and run them on different GPUs")
    parser.add_argument("--debug", action='store_true')
    
    args = parser.parse_args()

    set_seed()

    if args.combined:
        eval_mode_str = "combined"
    elif args.few_shot:
        eval_mode_str = "few_shot"
    elif args.just_instructions:
        eval_mode_str = "instructions"
    else:
        eval_mode_str = "basic"
    if args.parallel:
        eval_mode_str = eval_mode_str+"_parallel"
    else:
        eval_mode_str = eval_mode_str+"_serial"
    if args.debug:
        eval_mode_str = eval_mode_str + "_debug"

    run_name = args.model_name+"_"+eval_mode_str+"_"+MODEL_DESC

    log_file_name_base = run_name

    if args.save_dir:
        output_dir_root = os.path.join(args.save_dir, "results", run_name)
        visualize_dir_root = os.path.join(args.save_dir, "visualizations", run_name)
        log_file = os.path.join(args.save_dir, f"{log_file_name_base}.log")
        print(f"Using custom save directory: {args.save_dir}")
    else:
        output_dir_root = f"results_{run_name}"
        visualize_dir_root = f"visualizations_{run_name}"
        log_file = f"{log_file_name_base}.log"
        print(f"Using default save directories: {output_dir_root}, {visualize_dir_root}")

    os.makedirs(output_dir_root, exist_ok=True)
    os.makedirs(visualize_dir_root, exist_ok=True)
    if args.save_dir:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # for d in glob.glob(os.path.join(args.data_dir, "*")):
    #     print(os.path.basename(d))

    # all_dataset_dirs = sorted([d for d in glob.glob(os.path.join(args.data_dir, "*"))
    #                            if os.path.isdir(d) and os.path.exists(os.path.join(d, "test")) 
    #                            and (args.parallel and os.path.basename(d) in DATASET[args.cuda]) 
    #                            ])
    all_dataset_dirs = []

    for d in glob.glob(os.path.join(args.data_dir, "*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "test")):
            if args.parallel:
                if os.path.basename(d) in DATASET[args.cuda]:
                    all_dataset_dirs.append(d)
            else:
                all_dataset_dirs.append(d)
    all_dataset_dirs = sorted(all_dataset_dirs)

    logger.info(f"Found {len(all_dataset_dirs)} total datasets in {args.data_dir}. Will process all.")
    logger.info(f"Selected mode: {eval_mode_str}")
    logger.info(f"Using model: {args.model_name}")

    model, processor = load_qwen_model(args.model_name)

    dataset_stats = {}
    for dataset_dir in tqdm(all_dataset_dirs, desc="Processing datasets", unit="dataset"):
        dataset_name = os.path.basename(dataset_dir)
        logger.info(f"Starting processing dataset: {dataset_name}")
        _, _, processed, errored, skipped = process_dataset(
            model,
            processor,
            dataset_dir,
            args.few_shot,
            args.just_instructions,
            args.combined,
            output_dir_root,
            visualize_dir_root,
            args.model_name,
            args.debug,
        )
        dataset_stats[dataset_name] = {"processed": processed, "errors": errored, "skipped": skipped}
        logger.info(f"Finished processing dataset: {dataset_name}")

    logger.info("="*30 + " Processing Summary " + "="*30)
    total_processed_all = 0
    total_errors_all = 0
    total_skipped_all = 0
    successful_datasets = 0
    failed_datasets = 0

    for name, stats in dataset_stats.items():
        status_msg = stats.get("status", "")
        if "Skipped" in status_msg or "Failed" in status_msg:
             logger.warning(f"Dataset '{name}': {status_msg}")
             failed_datasets += 1
             continue

        processed = stats.get('processed', 0)
        errors = stats.get('errors', 0)
        skipped = stats.get('skipped', 0)

        total_processed_all += processed
        total_errors_all += errors
        total_skipped_all += skipped
        successful_datasets += 1

        logger.info(f"Dataset '{name}': Checked/Attempted {processed} images. Skipped: {skipped}. Errors: {errors}.")

    logger.info("-"*78)
    logger.info(f"Overall across {successful_datasets} successfully processed datasets (out of {len(dataset_stats)} total):")
    logger.info(f"Total image checks/attempts: {total_processed_all}")
    logger.info(f"Total images skipped (status True): {total_skipped_all}")
    logger.info(f"Total image-level errors/failures: {total_errors_all}")
    if failed_datasets > 0:
        logger.warning(f"{failed_datasets} datasets encountered critical errors or were skipped.")
    logger.info("="*78)
    if failed_datasets == 0:
        logger.info("All datasets processed.")
    else:
        logger.warning("Processing complete, but some datasets encountered errors.")


if __name__ == "__main__":
    main()